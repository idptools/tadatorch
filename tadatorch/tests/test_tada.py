"""
Tests for the user-facing predict() and predict_from_fasta() functions.
"""

import importlib.resources

import pytest

from tadatorch.backend.constants import TADA_SEQUENCE_LENGTH
from tadatorch.tada import predict, predict_from_fasta
from tadatorch.tests.reference_scores import TADATORCH_SCORES

SHORT_SEQUENCE: str = "EFSPENSSSSSWSSQESFLW"
EXACT_SEQUENCE: str = "EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD"
LONG_SEQUENCE: str = EXACT_SEQUENCE + EXACT_SEQUENCE

# Largest difference allowed between scores that should be the same.
SCORE_TOLERANCE: float = 0.000001

# seed used to make padding reproducible
RANDOM_SEED: int = 42


def get_test_fasta_path() -> str:
    """
    Get the path to the FASTA file shipped with tadatorch for testing.

    Returns
    -------
    str
        Path to a FASTA file holding four 40 amino acid sequences, named '0'
        to '3', which are the first four sequences in TADATORCH_SCORES.
    """
    return str(importlib.resources.files("tadatorch") / "data" / "testing.fasta")


def test_predict_single_sequence() -> None:
    """A single sequence can be passed as a string and comes back as the only key."""
    predictions = predict(EXACT_SEQUENCE)

    assert list(predictions.keys()) == [EXACT_SEQUENCE]
    assert len(predictions[EXACT_SEQUENCE]) == 1

    window_sequence, score = predictions[EXACT_SEQUENCE][0]
    assert window_sequence == EXACT_SEQUENCE
    assert isinstance(score, float)
    assert score == pytest.approx(TADATORCH_SCORES[EXACT_SEQUENCE], abs=SCORE_TOLERANCE)


def test_predict_long_sequence_is_windowed(capsys: pytest.CaptureFixture[str]) -> None:
    """An 80 amino acid sequence gives 41 windows with the default overlap of 39."""
    predictions = predict(LONG_SEQUENCE)

    # the user is warned that the sequence was windowed
    assert "Warning" in capsys.readouterr().out

    windows = [window_sequence for window_sequence, _ in predictions[LONG_SEQUENCE]]
    number_of_windows = len(LONG_SEQUENCE) - TADA_SEQUENCE_LENGTH + 1
    assert windows == [
        LONG_SEQUENCE[i : i + TADA_SEQUENCE_LENGTH] for i in range(number_of_windows)
    ]

    # the first and last windows are both the 40 amino acid test sequence
    for index in (0, -1):
        score = predictions[LONG_SEQUENCE][index][1]
        assert score == pytest.approx(TADATORCH_SCORES[EXACT_SEQUENCE], abs=SCORE_TOLERANCE)


def test_predict_overlap_length(capsys: pytest.CaptureFixture[str]) -> None:
    """With an overlap of 20, an 80 amino acid sequence gives 3 windows."""
    predictions = predict(LONG_SEQUENCE, overlap_length=20, verbose=False)

    # nothing should be printed when verbose is False
    assert capsys.readouterr().out == ""

    windows = [window_sequence for window_sequence, _ in predictions[LONG_SEQUENCE]]
    assert windows == [LONG_SEQUENCE[i : i + TADA_SEQUENCE_LENGTH] for i in (0, 20, 40)]


def test_predict_safe_mode_blocks_short_sequences() -> None:
    """Sequences under 40 amino acids should fail unless safe mode is turned off."""
    with pytest.raises(ValueError):
        predict(SHORT_SEQUENCE)
    with pytest.raises(ValueError):
        predict([EXACT_SEQUENCE, SHORT_SEQUENCE])


def test_predict_short_sequence_is_padded() -> None:
    """With safe mode off, a short sequence is padded to 40 amino acids."""
    predictions = predict(SHORT_SEQUENCE, safe_mode=False, verbose=False, approach="C")

    assert len(predictions[SHORT_SEQUENCE]) == 1
    padded_sequence, score = predictions[SHORT_SEQUENCE][0]

    assert isinstance(padded_sequence, str)
    assert len(padded_sequence) == TADA_SEQUENCE_LENGTH
    assert padded_sequence.startswith(SHORT_SEQUENCE)
    assert set(padded_sequence[len(SHORT_SEQUENCE) :]) <= {"G", "S"}
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_seed_makes_padding_reproducible() -> None:
    """The same seed should give the same padded sequence and the same score."""
    first = predict(SHORT_SEQUENCE, safe_mode=False, verbose=False, pad="random", seed=RANDOM_SEED)
    second = predict(SHORT_SEQUENCE, safe_mode=False, verbose=False, pad="random", seed=RANDOM_SEED)

    assert first == second


def test_predict_rejects_bad_sequences() -> None:
    """Empty input and sequences with invalid residues should fail."""
    with pytest.raises(ValueError):
        predict([])
    with pytest.raises(ValueError):
        predict(EXACT_SEQUENCE.lower())
    with pytest.raises(ValueError):
        predict("X" + EXACT_SEQUENCE[1:])


def test_predict_from_fasta() -> None:
    """Predictions from a FASTA file should be keyed by name and match predict()."""
    fasta_predictions = predict_from_fasta(get_test_fasta_path())

    expected_sequences = list(TADATORCH_SCORES.keys())[:4]
    assert list(fasta_predictions.keys()) == ["0", "1", "2", "3"]

    for name, expected_sequence in zip(fasta_predictions, expected_sequences):
        sequence, sequence_predictions = fasta_predictions[name]
        assert sequence == expected_sequence
        assert isinstance(sequence_predictions, list)

        window_sequence, score = sequence_predictions[0]
        assert window_sequence == expected_sequence
        assert score == pytest.approx(TADATORCH_SCORES[expected_sequence], abs=SCORE_TOLERANCE)


def test_predict_from_fasta_rejects_missing_file() -> None:
    """A path that does not exist should fail."""
    with pytest.raises(ValueError):
        predict_from_fasta("this_file_does_not_exist.fasta")
