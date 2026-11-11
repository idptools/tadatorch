"""
Command line interface for annotating the activation domains of the sequences
in a FASTA file.
"""

import argparse
import contextlib
import os
import sys
from typing import TextIO

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from tadatorch.backend.constants import TADA_SEQUENCE_LENGTH
from tadatorch.backend.domains import (
    DEFAULT_GAP_SIZE,
    DEFAULT_METHOD,
    DEFAULT_MIN_AD_SIZE,
    DEFAULT_SCORE_THRESHOLD,
    METHODS,
    ActivationDomain,
    calculate_residue_scores,
    find_activation_domains,
)
from tadatorch.plotting import open_profile_pdf, plot_sequence_profile
from tadatorch.tada import predict_window_scores, read_fasta_sequences

# Names of the columns in the TSV of activation domains, in the order they are
# written.
TSV_COLUMNS: tuple[str, ...] = (
    "name",
    "start",
    "end",
    "length",
    "windows_above_threshold",
    "mean_score",
    "max_score",
    "sequence",
)

# Names of the columns in the (optional) TSV of per-residue scores, in the
# order they are written.
RESIDUE_TSV_COLUMNS: tuple[str, ...] = (
    "name",
    "position",
    "residue",
    "score",
    "in_domain",
)

# Scores only reproduce between versions of TADA to around the sixth decimal
# place, so there is no information in writing out more digits than this.
SCORE_DECIMAL_PLACES: int = 4


def build_parser() -> argparse.ArgumentParser:
    """
    Build the parser for the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the tadatorch command.
    """
    parser = argparse.ArgumentParser(
        prog="tadatorch",
        description=(
            "Predict the transcriptional activation domains (ADs) of the sequences in a "
            "FASTA file and write them out as a tab-separated (TSV) file with one row "
            "per AD. A TAD score is predicted for every 40 amino acid window of each "
            "sequence. By default (--method whole_window) an AD is a contiguous region "
            "in which every residue is covered by at least one window with a score at "
            "or above the threshold. With --method central_residues only the two "
            "residues at the center of each window at or above the threshold are "
            "marked, gaps shorter than --gap-size are joined, ADs shorter than "
            "--min-ad-size are removed, and ADs at either end of a sequence are "
            "extended to the terminus if the --min-ad-size residues nearest that end "
            "that a window can mark were all marked. Start and end positions are "
            "1-indexed and inclusive."
        ),
    )
    parser.add_argument("fasta", help="Path to the FASTA file of protein sequences.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path the TSV is written to. If not set, the TSV is written to the terminal.",
    )
    parser.add_argument(
        "--per-residue",
        default=None,
        help=(
            "Also write a TSV with one row per residue to this path, giving each residue "
            "the highest score of the windows that mark it with the chosen --method "
            "(every window it is in with whole_window, or the windows it is at the "
            "center of with central_residues) and whether it is in an AD. Residues no "
            "window marks have a score of nan."
        ),
    )
    parser.add_argument(
        "--plot",
        default=None,
        help=(
            "Also write a PDF to this path with one page per sequence, showing the "
            "per-residue TAD score from the central_residues method and the disorder "
            "score predicted by metapredict V3, with the ADs (called with the chosen "
            "--method, as in the TSV) drawn as rectangles along the top."
        ),
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=(
            "Windows with a TAD score at or above this value (between 0 and 1) are "
            f"treated as ADs. Default is {DEFAULT_SCORE_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=METHODS,
        default=DEFAULT_METHOD,
        help=(
            "Which residues of a window at or above the threshold are marked as being "
            f"in an AD. Default is {DEFAULT_METHOD}."
        ),
    )
    parser.add_argument(
        "--gap-size",
        type=int,
        default=DEFAULT_GAP_SIZE,
        help=(
            "Gaps between ADs shorter than this many residues are joined. Only used "
            f"with --method central_residues. Default is {DEFAULT_GAP_SIZE}."
        ),
    )
    parser.add_argument(
        "--min-ad-size",
        type=int,
        default=DEFAULT_MIN_AD_SIZE,
        help=(
            "ADs shorter than this many residues are removed. This is also the number "
            "of residues at an end of a sequence that must be marked for an AD to be "
            "extended to that terminus. Only used with --method central_residues. "
            f"Default is {DEFAULT_MIN_AD_SIZE}."
        ),
    )
    return parser


def format_domain_row(name: str, domain: ActivationDomain) -> str:
    """
    Format an activation domain as one row of the TSV.

    Parameters
    ----------
    name : str
        Name of the sequence the domain was found in.

    domain : ActivationDomain
        The activation domain.

    Returns
    -------
    str
        The tab-separated row (without a newline), with values in the order
        of TSV_COLUMNS.
    """
    values = [
        name,
        str(domain.start),
        str(domain.end),
        str(len(domain.sequence)),
        str(domain.number_of_windows_above_threshold),
        f"{domain.mean_score:.{SCORE_DECIMAL_PLACES}f}",
        f"{domain.max_score:.{SCORE_DECIMAL_PLACES}f}",
        domain.sequence,
    ]
    assert len(values) == len(TSV_COLUMNS), "A value is needed for every TSV column."

    return "\t".join(values)


def mark_domain_residues(sequence_length: int, domains: list[ActivationDomain]) -> np.ndarray:
    """
    Mark which residues of a sequence are in one of its activation domains.

    Parameters
    ----------
    sequence_length : int
        The number of residues in the sequence.

    domains : list of ActivationDomain
        The activation domains found in the sequence.

    Returns
    -------
    np.ndarray
        Boolean array with one value per residue, which is True for residues
        inside an activation domain.
    """
    is_in_domain = np.zeros(sequence_length, dtype=bool)

    # start and end are 1-indexed and inclusive
    for domain in domains:
        is_in_domain[domain.start - 1 : domain.end] = True

    return is_in_domain


def format_residue_rows(
    name: str, sequence: str, residue_scores: np.ndarray, domains: list[ActivationDomain]
) -> list[str]:
    """
    Format the per-residue scores of a sequence as rows of the per-residue TSV.

    Parameters
    ----------
    name : str
        Name of the sequence.

    sequence : str
        Amino acid sequence.

    residue_scores : np.ndarray
        One score per residue, as returned by ``calculate_residue_scores()``.
        Residues no window marks are NaN, and are written as 'nan'.

    domains : list of ActivationDomain
        The activation domains found in the sequence, used to fill in the
        in_domain column.

    Returns
    -------
    list of str
        One tab-separated row (without a newline) per residue, in order, with
        values in the order of RESIDUE_TSV_COLUMNS. Positions are 1-indexed,
        and in_domain is 1 for residues in an activation domain and 0
        otherwise.
    """
    assert residue_scores.shape == (len(sequence),), (
        f"Got {residue_scores.shape} residue scores for a sequence of {len(sequence)} residues."
    )

    is_in_domain = mark_domain_residues(len(sequence), domains)

    rows = []
    for index, residue in enumerate(sequence):
        values = [
            name,
            str(index + 1),
            residue,
            f"{residue_scores[index]:.{SCORE_DECIMAL_PLACES}f}",
            str(int(is_in_domain[index])),
        ]
        rows.append("\t".join(values))

    return rows


def check_names_are_safe_for_tsv(names: list[str]) -> None:
    """
    Check that no sequence name contains a tab.

    Parameters
    ----------
    names : list of str
        Names of the sequences read from the FASTA file.

    Raises
    ------
    ValueError
        If any name contains a tab, as this would add extra columns to that
        sequence's rows in the TSV.
    """
    names_with_tabs = [name for name in names if "\t" in name]
    if len(names_with_tabs) > 0:
        raise ValueError(
            f"{len(names_with_tabs)} FASTA header(s) contain a tab, which would corrupt "
            f"the TSV. The first is: {names_with_tabs[0]!r}"
        )


def add_profile_page(
    plot_pdf: PdfPages,
    name: str,
    sequence: str,
    window_scores: np.ndarray,
    domains: list[ActivationDomain],
) -> None:
    """
    Plot the per-residue profile of a sequence (see
    ``plot_sequence_profile()``) as a new page of a PDF.

    Parameters
    ----------
    plot_pdf : PdfPages
        The open PDF the page is added to.

    name : str
        Name of the sequence.

    sequence : str
        Amino acid sequence.

    window_scores : np.ndarray
        TAD score for every 40 amino acid window of the sequence.

    domains : list of ActivationDomain
        The activation domains found in the sequence, drawn on the domain map.
    """
    figure = plot_sequence_profile(name, sequence, window_scores, domains)
    # matplotlib does not type-annotate PdfPages.savefig
    plot_pdf.savefig(figure)  # type: ignore[no-untyped-call]


def write_activation_domains(
    names_to_sequences: dict[str, str],
    threshold: float,
    method: str,
    gap_size: int,
    min_ad_size: int,
    output_file: TextIO,
    log_file: TextIO,
    residue_file: TextIO | None = None,
    plot_pdf: PdfPages | None = None,
) -> None:
    """
    Predict the activation domains of every sequence and write them as a TSV,
    and optionally write a second TSV of per-residue scores and a PDF of
    per-residue profiles.

    Sequences under 40 amino acids cannot be scored by TADA, so are skipped.
    The number (and names) of any skipped sequences are reported in the log.

    Parameters
    ----------
    names_to_sequences : dict
        Dictionary where each key is the name of a sequence and each value is
        the sequence.

    threshold : float
        Windows with a TAD score at or above this value (between 0 and 1) are
        treated as activation domains.

    method : str
        Which residues of a window are marked: 'whole_window' or
        'central_residues'.

    gap_size : int
        Gaps between domains shorter than this many residues are joined (only
        used by the 'central_residues' method).

    min_ad_size : int
        Domains shorter than this many residues are removed (only used by the
        'central_residues' method).

    output_file : TextIO
        Open file the TSV is written to.

    log_file : TextIO
        Open file that progress and summary messages are written to.

    residue_file : TextIO or None
        Open file the per-residue TSV is written to (see
        ``format_residue_rows()``). If None, no per-residue scores are
        written. Default is None.

    plot_pdf : PdfPages or None
        Open PDF that the profile of each sequence is added to as a new page
        (see ``add_profile_page()``). If None, no profiles are plotted.
        Default is None.
    """
    sequences_to_predict = {
        name: sequence
        for name, sequence in names_to_sequences.items()
        if len(sequence) >= TADA_SEQUENCE_LENGTH
    }
    skipped_names = [name for name in names_to_sequences if name not in sequences_to_predict]

    output_file.write("\t".join(TSV_COLUMNS) + "\n")
    if residue_file is not None:
        residue_file.write("\t".join(RESIDUE_TSV_COLUMNS) + "\n")

    number_of_domains = 0
    number_of_sequences_with_domains = 0

    for sequence_number, (name, sequence) in enumerate(sequences_to_predict.items(), start=1):
        # the windows are scored once, and those scores are used for the
        # activation domains, the per-residue scores and the plot
        window_scores = predict_window_scores(sequence)
        domains = find_activation_domains(
            sequence,
            window_scores,
            threshold=threshold,
            method=method,
            gap_size=gap_size,
            min_ad_size=min_ad_size,
        )

        # rows are written as soon as each sequence is finished, so a long run
        # that gets interrupted still leaves the results so far
        for domain in domains:
            output_file.write(format_domain_row(name, domain) + "\n")
        output_file.flush()

        if residue_file is not None:
            residue_scores = calculate_residue_scores(sequence, window_scores, method)
            for row in format_residue_rows(name, sequence, residue_scores, domains):
                residue_file.write(row + "\n")
            residue_file.flush()

        if plot_pdf is not None:
            add_profile_page(plot_pdf, name, sequence, window_scores, domains)

        number_of_domains += len(domains)
        if len(domains) > 0:
            number_of_sequences_with_domains += 1

        print(
            f"[{sequence_number}/{len(sequences_to_predict)}] {name}: {len(domains)} AD(s)",
            file=log_file,
        )

    print(
        f"Found {number_of_domains} AD(s) in {number_of_sequences_with_domains} of "
        f"{len(sequences_to_predict)} sequence(s) using a threshold of {threshold} "
        f"and the {method} method.",
        file=log_file,
    )
    if len(skipped_names) > 0:
        print(
            f"Skipped {len(skipped_names)} sequence(s) under {TADA_SEQUENCE_LENGTH} "
            f"amino acids: {', '.join(skipped_names)}",
            file=log_file,
        )


def main(argv: list[str] | None = None) -> None:
    """
    Run the tadatorch command line interface.

    Parameters
    ----------
    argv : list of str or None
        Command line arguments. If None the arguments are taken from
        ``sys.argv``. Default is None.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if not 0 <= arguments.threshold <= 1:
        parser.error(f"--threshold must be between 0 and 1, but got {arguments.threshold}.")
    if arguments.gap_size < 0 or arguments.min_ad_size < 0:
        parser.error("--gap-size and --min-ad-size cannot be negative.")
    output_paths = [
        path
        for path in (arguments.output, arguments.per_residue, arguments.plot)
        if path is not None
    ]
    if len({os.path.realpath(path) for path in output_paths}) < len(output_paths):
        parser.error("--output, --per-residue and --plot must all be different files.")
    if arguments.plot is not None and not arguments.plot.lower().endswith(".pdf"):
        parser.error(
            f"--plot writes a PDF, so its path must end in .pdf, but got {arguments.plot}."
        )

    names_to_sequences = read_fasta_sequences(arguments.fasta)
    if len(names_to_sequences) == 0:
        raise ValueError(f"No sequences were found in {arguments.fasta}")
    check_names_are_safe_for_tsv(list(names_to_sequences.keys()))

    settings = (arguments.threshold, arguments.method, arguments.gap_size, arguments.min_ad_size)

    # each output file may or may not be opened, and the ExitStack closes
    # whichever ones were
    with contextlib.ExitStack() as open_files:
        output_file: TextIO = sys.stdout
        if arguments.output is not None:
            output_file = open_files.enter_context(open(arguments.output, "w"))

        residue_file: TextIO | None = None
        if arguments.per_residue is not None:
            residue_file = open_files.enter_context(open(arguments.per_residue, "w"))

        plot_pdf: PdfPages | None = None
        if arguments.plot is not None:
            plot_pdf = open_files.enter_context(open_profile_pdf(arguments.plot))

        # progress goes to stderr so it never ends up mixed in with a TSV that
        # is being written to stdout
        write_activation_domains(
            names_to_sequences,
            *settings,
            output_file=output_file,
            log_file=sys.stderr,
            residue_file=residue_file,
            plot_pdf=plot_pdf,
        )


if __name__ == "__main__":
    main()
