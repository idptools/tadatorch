"""
Command line interface for annotating the activation domains of the sequences
in a FASTA file.
"""

import argparse
import sys
from typing import TextIO

from tadatorch.backend.constants import TADA_SEQUENCE_LENGTH
from tadatorch.backend.domains import (
    DEFAULT_GAP_SIZE,
    DEFAULT_METHOD,
    DEFAULT_MIN_AD_SIZE,
    DEFAULT_SCORE_THRESHOLD,
    METHODS,
    ActivationDomain,
)
from tadatorch.tada import predict_activation_domains, read_fasta_sequences

# Names of the columns in the TSV, in the order they are written.
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


def write_activation_domains(
    names_to_sequences: dict[str, str],
    threshold: float,
    method: str,
    gap_size: int,
    min_ad_size: int,
    output_file: TextIO,
    log_file: TextIO,
) -> None:
    """
    Predict the activation domains of every sequence and write them as a TSV.

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
    """
    sequences_to_predict = {
        name: sequence
        for name, sequence in names_to_sequences.items()
        if len(sequence) >= TADA_SEQUENCE_LENGTH
    }
    skipped_names = [name for name in names_to_sequences if name not in sequences_to_predict]

    output_file.write("\t".join(TSV_COLUMNS) + "\n")

    number_of_domains = 0
    number_of_sequences_with_domains = 0

    for sequence_number, (name, sequence) in enumerate(sequences_to_predict.items(), start=1):
        domains = predict_activation_domains(
            sequence,
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

    names_to_sequences = read_fasta_sequences(arguments.fasta)
    if len(names_to_sequences) == 0:
        raise ValueError(f"No sequences were found in {arguments.fasta}")
    check_names_are_safe_for_tsv(list(names_to_sequences.keys()))

    # progress goes to stderr so it never ends up mixed in with a TSV that is
    # being written to stdout
    settings = (arguments.threshold, arguments.method, arguments.gap_size, arguments.min_ad_size)

    if arguments.output is None:
        write_activation_domains(
            names_to_sequences, *settings, output_file=sys.stdout, log_file=sys.stderr
        )
        return

    with open(arguments.output, "w") as output_file:
        write_activation_domains(
            names_to_sequences, *settings, output_file=output_file, log_file=sys.stderr
        )


if __name__ == "__main__":
    main()
