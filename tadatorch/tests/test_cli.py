"""
Tests for the command line interface.
"""

from pathlib import Path

import pytest

from tadatorch.backend.domains import ActivationDomain
from tadatorch.cli import TSV_COLUMNS, format_domain_row, main
from tadatorch.tests.reference_scores import TADATORCH_SCORES

# A 40 amino acid sequence that scores above 0.5 (about 0.648).
ACTIVE_SEQUENCE: str = "EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD"

# A 40 amino acid sequence that scores below 0.5 (about 0.205).
INACTIVE_SEQUENCE: str = "EVTKSQSFDHPQPDIPCGFEDTNEESDLRRQLVESTTPNN"

SHORT_SEQUENCE: str = "EFSPENSSSSSWSSQESFLW"


def write_fasta(path: Path, names_to_sequences: dict[str, str]) -> str:
    """
    Write sequences to a FASTA file.

    Parameters
    ----------
    path : Path
        Where the FASTA file is written.

    names_to_sequences : dict
        Dictionary where each key is the name of a sequence and each value is
        the sequence.

    Returns
    -------
    str
        The path to the FASTA file.
    """
    lines = [f">{name}\n{sequence}\n" for name, sequence in names_to_sequences.items()]
    path.write_text("".join(lines))
    return str(path)


def read_tsv(text: str) -> list[dict[str, str]]:
    """
    Split the text of a TSV into one dictionary per row.

    Parameters
    ----------
    text : str
        The contents of a TSV with a header row.

    Returns
    -------
    list of dict
        A dictionary for each row after the header, mapping column names to
        the (string) values in that row.
    """
    lines = text.splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_format_domain_row() -> None:
    """A row should have one value per column, with scores to 4 decimal places."""
    domain = ActivationDomain(
        start=11,
        end=50,
        sequence="A" * 40,
        number_of_windows_above_threshold=3,
        mean_score=0.61234567,
        max_score=0.7,
    )

    assert format_domain_row("my protein", domain).split("\t") == [
        "my protein",
        "11",
        "50",
        "40",
        "3",
        "0.6123",
        "0.7000",
        "A" * 40,
    ]


def test_cli_writes_tsv_to_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Run the whole command on a small FASTA file. Only the sequence with a
    score over the threshold should get a row, and the sequence under 40
    amino acids should be reported as skipped.
    """
    fasta_path = write_fasta(
        tmp_path / "input.fasta",
        {
            "active protein": ACTIVE_SEQUENCE,
            "inactive": INACTIVE_SEQUENCE,
            "short": SHORT_SEQUENCE,
        },
    )
    output_path = tmp_path / "output.tsv"

    main([fasta_path, "--output", str(output_path)])

    output_text = output_path.read_text()
    assert output_text.splitlines()[0].split("\t") == list(TSV_COLUMNS)

    rows = read_tsv(output_text)
    assert len(rows) == 1
    row = rows[0]

    assert row["name"] == "active protein"
    assert (row["start"], row["end"], row["length"]) == ("1", "40", "40")
    assert row["windows_above_threshold"] == "1"
    assert row["sequence"] == ACTIVE_SEQUENCE
    assert float(row["max_score"]) == pytest.approx(TADATORCH_SCORES[ACTIVE_SEQUENCE], abs=1e-4)
    assert row["mean_score"] == row["max_score"]

    # nothing is written to stdout when an output file is given, and the
    # summary (including the skipped sequence) goes to stderr
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Found 1 AD(s) in 1 of 2 sequence(s)" in captured.err
    assert "Skipped 1 sequence(s) under 40 amino acids: short" in captured.err


def test_cli_writes_tsv_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without an output path the TSV should be written to stdout."""
    fasta_path = write_fasta(tmp_path / "input.fasta", {"active": ACTIVE_SEQUENCE})

    main([fasta_path])

    rows = read_tsv(capsys.readouterr().out)
    assert [row["name"] for row in rows] == ["active"]


def test_cli_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A lower threshold should let the lower scoring sequence through."""
    fasta_path = write_fasta(
        tmp_path / "input.fasta", {"active": ACTIVE_SEQUENCE, "inactive": INACTIVE_SEQUENCE}
    )

    main([fasta_path, "--threshold", "0.2"])

    rows = read_tsv(capsys.readouterr().out)
    assert [row["name"] for row in rows] == ["active", "inactive"]


def test_cli_central_residues_method(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    With the central residues method the one window of a 40 amino acid
    sequence only marks its two central residues, which is under the default
    minimum AD size. With a minimum AD size of 2 those two residues are
    enough to keep the AD and to extend it to both ends of the sequence.
    """
    fasta_path = write_fasta(tmp_path / "input.fasta", {"active": ACTIVE_SEQUENCE})

    main([fasta_path, "--method", "central_residues"])
    assert read_tsv(capsys.readouterr().out) == []

    main([fasta_path, "--method", "central_residues", "--min-ad-size", "2"])
    rows = read_tsv(capsys.readouterr().out)
    assert [(row["start"], row["end"], row["sequence"]) for row in rows] == [
        ("1", "40", ACTIVE_SEQUENCE)
    ]


def test_cli_rejects_bad_input(tmp_path: Path) -> None:
    """Missing files, empty files, tabs in names and bad thresholds should fail."""
    with pytest.raises(ValueError):
        main([str(tmp_path / "missing.fasta")])

    empty_path = tmp_path / "empty.fasta"
    empty_path.write_text("")
    with pytest.raises(ValueError):
        main([str(empty_path)])

    tab_path = write_fasta(tmp_path / "tab.fasta", {"name\twith tab": ACTIVE_SEQUENCE})
    with pytest.raises(ValueError):
        main([tab_path])

    # argparse reports bad arguments by exiting
    fasta_path = write_fasta(tmp_path / "input.fasta", {"active": ACTIVE_SEQUENCE})
    with pytest.raises(SystemExit):
        main([fasta_path, "--threshold", "1.5"])
