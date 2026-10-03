"""
Tests for the sequence padding, windowing and validation utilities.
"""

import random

import pytest

from tadatorch.backend.constants import AMINO_ACIDS, TADA_SEQUENCE_LENGTH
from tadatorch.backend.utils import (
    make_sequences_constant_length,
    map_sequences_to_prediction,
    pad_sequence,
    sliding_window,
    validate_sequence,
    verbose_warning_message,
)

SHORT_SEQUENCE: str = "EFSPENSSSSSWSSQESFLW"
EXACT_SEQUENCE: str = "EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD"
LONG_SEQUENCE: str = EXACT_SEQUENCE + "EF"

# seed used to make padding reproducible
RANDOM_SEED: int = 42


def test_validate_sequence() -> None:
    """Only uppercase sequences of the 20 standard amino acids should pass."""
    validate_sequence("".join(AMINO_ACIDS))

    with pytest.raises(ValueError):
        validate_sequence("EFSPX")
    with pytest.raises(ValueError):
        validate_sequence("efspe")
    with pytest.raises(ValueError):
        validate_sequence(["EFSPE"])  # type: ignore[arg-type]


def test_sliding_window() -> None:
    """Windows should be the right length and step by window_length - overlap."""
    assert sliding_window("ABCDEFGH", window_length=4, overlap=3) == [
        "ABCD",
        "BCDE",
        "CDEF",
        "DEFG",
        "EFGH",
    ]
    assert sliding_window("ABCDEFGH", window_length=4, overlap=0) == ["ABCD", "EFGH"]

    # residues left over at the C-terminus are not part of any window
    assert sliding_window("ABCDEFGHI", window_length=4, overlap=0) == ["ABCD", "EFGH"]


def test_sliding_window_rejects_bad_arguments() -> None:
    """Window lengths and overlaps that make no sense should fail."""
    with pytest.raises(ValueError):
        sliding_window("ABCD", window_length=5, overlap=0)
    with pytest.raises(ValueError):
        sliding_window("ABCD", window_length=0, overlap=0)
    with pytest.raises(ValueError):
        sliding_window("ABCD", window_length=2, overlap=2)
    with pytest.raises(ValueError):
        sliding_window("ABCD", window_length=2, overlap=-1)


@pytest.mark.parametrize("approach", ["even", "N", "C"])
def test_pad_sequence_position(approach: str) -> None:
    """Padding should bring the sequence to 40 residues and go on the right end(s)."""
    padded_sequence = pad_sequence(SHORT_SEQUENCE, approach=approach)
    number_padded = TADA_SEQUENCE_LENGTH - len(SHORT_SEQUENCE)

    assert len(padded_sequence) == TADA_SEQUENCE_LENGTH

    if approach == "N":
        assert padded_sequence.endswith(SHORT_SEQUENCE)
    elif approach == "C":
        assert padded_sequence.startswith(SHORT_SEQUENCE)
    else:
        number_n_terminal = number_padded // 2
        assert padded_sequence[number_n_terminal:].startswith(SHORT_SEQUENCE)


def test_pad_sequence_residues() -> None:
    """'GS' should only pad with G and S; 'random' can use any amino acid."""
    gs_padded = pad_sequence(SHORT_SEQUENCE, pad="GS", approach="N")
    gs_padding = gs_padded[: -len(SHORT_SEQUENCE)]
    assert set(gs_padding) <= {"G", "S"}

    random_padded = pad_sequence(SHORT_SEQUENCE, pad="random", approach="N")
    random_padding = random_padded[: -len(SHORT_SEQUENCE)]
    assert set(random_padding) <= set(AMINO_ACIDS)


def test_pad_sequence_is_reproducible_with_seeded_generator() -> None:
    """The same seeded generator should always give the same padding."""
    first = pad_sequence(SHORT_SEQUENCE, pad="random", random_generator=random.Random(RANDOM_SEED))
    second = pad_sequence(SHORT_SEQUENCE, pad="random", random_generator=random.Random(RANDOM_SEED))

    assert first == second


def test_pad_sequence_leaves_long_sequences_alone() -> None:
    """Sequences at or over the objective length should be returned unchanged."""
    assert pad_sequence(EXACT_SEQUENCE) == EXACT_SEQUENCE
    assert pad_sequence(LONG_SEQUENCE) == LONG_SEQUENCE


def test_pad_sequence_rejects_bad_arguments() -> None:
    """Invalid padding options should fail."""
    with pytest.raises(ValueError):
        pad_sequence(SHORT_SEQUENCE, pad="X")
    with pytest.raises(ValueError):
        pad_sequence(SHORT_SEQUENCE, approach="middle")
    with pytest.raises(ValueError):
        pad_sequence(SHORT_SEQUENCE, objective_length=-1)


def test_make_sequences_constant_length() -> None:
    """Every input sequence should map to a list of 40 amino acid sequences."""
    sequences = [LONG_SEQUENCE, SHORT_SEQUENCE, EXACT_SEQUENCE]
    sequence_dict = make_sequences_constant_length(sequences, seed=RANDOM_SEED)

    # keys are the input sequences, in the order they were passed
    assert list(sequence_dict.keys()) == sequences

    # a 42 amino acid sequence has 3 windows of 40 when the overlap is 39
    assert sequence_dict[LONG_SEQUENCE] == [LONG_SEQUENCE[i : i + 40] for i in range(3)]
    assert sequence_dict[EXACT_SEQUENCE] == [EXACT_SEQUENCE]

    assert len(sequence_dict[SHORT_SEQUENCE]) == 1
    assert SHORT_SEQUENCE in sequence_dict[SHORT_SEQUENCE][0]

    for constant_length_sequences in sequence_dict.values():
        assert all(len(sequence) == TADA_SEQUENCE_LENGTH for sequence in constant_length_sequences)


def test_make_sequences_constant_length_is_reproducible_with_seed() -> None:
    """The same seed should give the same padded sequences."""
    first = make_sequences_constant_length([SHORT_SEQUENCE], pad="random", seed=RANDOM_SEED)
    second = make_sequences_constant_length([SHORT_SEQUENCE], pad="random", seed=RANDOM_SEED)

    assert first == second


def test_map_sequences_to_prediction() -> None:
    """The flat list and the indices should let us recover the original mapping."""
    sequence_dict = {"first": ["a", "b", "c"], "second": ["d"], "third": ["e", "f"]}
    all_sequences, sequence_to_indices = map_sequences_to_prediction(sequence_dict)

    assert all_sequences == ["a", "b", "c", "d", "e", "f"]
    assert sequence_to_indices == {"first": [0, 1, 2], "second": [3], "third": [4, 5]}


def test_verbose_warning_message() -> None:
    """The warning should describe the padding and windowing that will be used."""
    message = verbose_warning_message(overlap_length=39, pad="GS", approach="N")

    assert "random selection of G and S" in message
    assert "only at the N terminus" in message
    assert "39 overlapping amino acids" in message
