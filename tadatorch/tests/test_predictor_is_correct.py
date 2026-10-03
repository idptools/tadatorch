"""
Makes sure the predictor gets the scores it should, and that those scores stay
close to the ones from previous versions of TADA.
"""

import numpy as np
import pytest

from tadatorch.backend.predictor import predict_tada
from tadatorch.tada import predict
from tadatorch.tests.reference_scores import (
    ORIGINAL_TADA_SCORES,
    TADA_T2_README_SCORES,
    TADATORCH_SCORES,
)

# Largest difference allowed between scores that should be the same.
SCORE_TOLERANCE: float = 0.000001

# Largest difference allowed between a tadatorch score and the score from a
# previous version of TADA. The scores are not identical because tadatorch
# calculates kappa and Omega with sparrow rather than localcider. The largest
# difference across the reference sequences is 0.03.
PREVIOUS_VERSION_TOLERANCE: float = 0.05

# Most scores are much closer than that to previous versions of TADA, so the
# median difference is also checked.
PREVIOUS_VERSION_MEDIAN_TOLERANCE: float = 0.01


def test_predictor_reproduces_tadatorch_scores() -> None:
    """
    Make sure the scores tadatorch predicts have not changed. This tests 100
    sequences.
    """
    sequences = list(TADATORCH_SCORES.keys())
    tada_scores = predict(sequences)

    for sequence in sequences:
        tadatorch_score = tada_scores[sequence][0][1]
        assert tadatorch_score == pytest.approx(TADATORCH_SCORES[sequence], abs=SCORE_TOLERANCE)


@pytest.mark.parametrize(
    "reference_scores",
    [ORIGINAL_TADA_SCORES, TADA_T2_README_SCORES],
    ids=["original TADA", "TADA_T2 README"],
)
def test_predictor_is_close_to_previous_versions(reference_scores: dict[str, float]) -> None:
    """
    Make sure tadatorch stays close to the TAD scores from the original
    version of TADA (100 sequences) and those reported in the TADA_T2 README
    (44 sequences).
    """
    sequences = list(reference_scores.keys())
    tada_scores = predict(sequences)

    differences = np.array(
        [float(tada_scores[sequence][0][1]) - reference_scores[sequence] for sequence in sequences]
    )

    assert np.abs(differences).max() < PREVIOUS_VERSION_TOLERANCE
    assert np.median(np.abs(differences)) < PREVIOUS_VERSION_MEDIAN_TOLERANCE


def test_return_both_values() -> None:
    """
    Make sure both network outputs are returned when asked for, and that the
    first is the TAD score and the second is 1 - TAD score.
    """
    sequences = list(ORIGINAL_TADA_SCORES.keys())[:5]

    tad_scores = predict_tada(sequences)
    both_values = predict_tada(sequences, return_both_values=True)

    assert isinstance(both_values, np.ndarray)
    assert both_values.shape == (len(sequences), 2)
    np.testing.assert_allclose(both_values[:, 0], tad_scores)
    np.testing.assert_allclose(both_values.sum(axis=1), 1.0, atol=SCORE_TOLERANCE)


def test_batch_size_does_not_change_predictions() -> None:
    """Make sure the scores do not depend on how sequences are batched."""
    sequences = list(ORIGINAL_TADA_SCORES.keys())[:7]

    one_batch = predict_tada(sequences, batch_size=len(sequences))
    many_batches = predict_tada(sequences, batch_size=2)

    np.testing.assert_allclose(one_batch, many_batches, atol=SCORE_TOLERANCE)


def test_predict_tada_rejects_bad_input() -> None:
    """Make sure the backend predictor fails loudly on input it cannot use."""
    sequence = next(iter(ORIGINAL_TADA_SCORES))

    with pytest.raises(ValueError):
        predict_tada(sequence)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        predict_tada([sequence], batch_size=0)
    with pytest.raises(ValueError):
        predict_tada([sequence[:-1]])
