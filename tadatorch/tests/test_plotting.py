"""
Tests for plotting per-residue profiles.
"""

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from tadatorch.backend.domains import ActivationDomain
from tadatorch.plotting import (
    AD_FACE_COLOR,
    PLOT_RC_PARAMS,
    plot_residue_profile,
    predict_disorder,
    residue_axis_limits,
)

TEST_SEQUENCE: str = "EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD"

# A sequence length and two activation domains used to build profiles by hand.
PROFILE_LENGTH: int = 100
PROFILE_DOMAINS: list[ActivationDomain] = [
    ActivationDomain(
        start=10,
        end=29,
        sequence="A" * 20,
        number_of_windows_above_threshold=1,
        mean_score=0.6,
        max_score=0.6,
    ),
    ActivationDomain(
        start=60,
        end=60,
        sequence="A",
        number_of_windows_above_threshold=1,
        mean_score=0.7,
        max_score=0.7,
    ),
]


def make_test_profile() -> tuple[np.ndarray, np.ndarray]:
    """
    Make per-residue TAD and disorder scores for a sequence of PROFILE_LENGTH
    residues.

    Returns
    -------
    np.ndarray
        TAD scores that rise from 0 to 1 along the sequence, with the first
        and last 19 residues NaN (as the central residues method gives).

    np.ndarray
        Disorder scores that fall from 1 to 0 along the sequence.
    """
    central_residue_scores = np.linspace(0.0, 1.0, PROFILE_LENGTH)
    central_residue_scores[:19] = np.nan
    central_residue_scores[-19:] = np.nan
    disorder_scores = np.linspace(1.0, 0.0, PROFILE_LENGTH)
    return central_residue_scores, disorder_scores


def test_predict_disorder() -> None:
    """There should be one disorder score per residue, between 0 and 1."""
    disorder_scores = predict_disorder(TEST_SEQUENCE)

    assert disorder_scores.shape == (len(TEST_SEQUENCE),)
    assert np.all((disorder_scores >= 0.0) & (disorder_scores <= 1.0))


def test_profile_has_a_domain_map_above_the_scores() -> None:
    """
    The figure should have the domain map on top and the score panel below,
    sharing a residue axis where each residue takes up one unit.
    """
    central_residue_scores, disorder_scores = make_test_profile()
    figure = plot_residue_profile(
        "my protein", central_residue_scores, disorder_scores, PROFILE_DOMAINS
    )

    domain_map_axes, score_axes = figure.axes
    assert domain_map_axes.get_position().y0 > score_axes.get_position().y0
    assert score_axes.get_xlim() == residue_axis_limits(PROFILE_LENGTH)
    assert domain_map_axes.get_xlim() == score_axes.get_xlim()

    assert domain_map_axes.get_title(loc="left") == "my protein"
    assert score_axes.get_xlabel() == "Residue"
    assert score_axes.get_ylabel() == "Score"
    assert score_axes.xaxis.label.get_fontfamily() == [PLOT_RC_PARAMS["font.family"]]


def test_profile_draws_each_domain_as_a_rectangle() -> None:
    """Each domain should cover its residues from half a residue either side."""
    central_residue_scores, disorder_scores = make_test_profile()
    figure = plot_residue_profile("name", central_residue_scores, disorder_scores, PROFILE_DOMAINS)

    rectangles = [patch for patch in figure.axes[0].patches if isinstance(patch, Rectangle)]
    extents = [
        (rectangle.get_x(), rectangle.get_x() + rectangle.get_width()) for rectangle in rectangles
    ]

    assert extents == [(9.5, 29.5), (59.5, 60.5)]
    for rectangle in rectangles:
        assert rectangle.get_facecolor() == to_rgba(AD_FACE_COLOR)


def test_profile_plots_both_scores_at_every_residue() -> None:
    """The score panel should have one line per score, with a point per residue."""
    central_residue_scores, disorder_scores = make_test_profile()
    figure = plot_residue_profile("name", central_residue_scores, disorder_scores, PROFILE_DOMAINS)

    lines = [line for line in figure.axes[1].get_lines() if isinstance(line, Line2D)]
    assert [line.get_label() for line in lines] == ["TAD score", "Disorder"]

    tad_line, disorder_line = lines
    np.testing.assert_array_equal(tad_line.get_xdata(), np.arange(1, PROFILE_LENGTH + 1))
    np.testing.assert_array_equal(tad_line.get_ydata(), central_residue_scores)
    np.testing.assert_array_equal(disorder_line.get_ydata(), disorder_scores)
    assert figure.axes[1].get_ylim() == (0.0, 1.0)


def test_profile_without_domains_only_draws_the_sequence_line() -> None:
    """A sequence with no domains should still get a domain map line."""
    central_residue_scores, disorder_scores = make_test_profile()
    figure = plot_residue_profile("name", central_residue_scores, disorder_scores, [])

    domain_map_axes = figure.axes[0]
    assert len(domain_map_axes.patches) == 0
    assert len(domain_map_axes.get_lines()) == 1
