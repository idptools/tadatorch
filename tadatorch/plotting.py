"""
Code for plotting per-residue profiles of TAD scores, predicted disorder and
activation domains.

Each profile is one figure with two panels that share the residue axis. The
bottom panel shows the per-residue TAD score from the 'central_residues'
method and the disorder score predicted by metapredict (both run from 0 to
1). The top panel is a domain map with the activation domains drawn as
rectangles along a line the length of the sequence.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import matplotlib
import metapredict
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.typing import RcKeyType

from tadatorch.backend.domains import (
    CENTRAL_RESIDUES_METHOD,
    ActivationDomain,
    calculate_residue_scores,
)

# The metapredict network used to predict disorder. metapredict is described in
# Emenecker, Griffith & Holehouse 2021, Biophys. J. 120, 4312-4319, and its
# networks (including V3) at https://github.com/idptools/metapredict.
METAPREDICT_VERSION: str = "V3"

# Plot settings. Type 42 (TrueType) fonts are embedded as text rather than as
# outlines, so text in the PDF can be edited in Illustrator.
PLOT_RC_PARAMS: dict[RcKeyType, Any] = {
    "font.family": "Arial",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Colors follow the default xmgrace order (black, red, blue).
TAD_SCORE_COLOR: str = "black"
DISORDER_COLOR: str = "red"
AD_FACE_COLOR: str = "blue"
LINE_COLOR: str = "black"

# Figure size in inches (width, height), and the relative heights of the
# domain map and the score panel.
FIGURE_SIZE_INCHES: tuple[float, float] = (7.0, 2.6)
PANEL_HEIGHT_RATIOS: tuple[int, int] = (1, 5)

# Line widths in points.
PROFILE_LINE_WIDTH: float = 1.0
DOMAIN_MAP_LINE_WIDTH: float = 1.0
AD_EDGE_LINE_WIDTH: float = 0.5

# In the domain map (which runs from 0 to 1 vertically), the line showing the
# sequence sits in the middle and each activation domain is a rectangle of
# this height centered on it. The rectangles are drawn on top of the line.
DOMAIN_MAP_CENTER: float = 0.5
AD_RECTANGLE_HEIGHT: float = 0.8
SEQUENCE_LINE_ZORDER: int = 1
AD_RECTANGLE_ZORDER: int = 2

# Both scores run from 0 to 1.
SCORE_AXIS_LIMITS: tuple[float, float] = (0.0, 1.0)

# Font size (in points) of the sequence name written above the domain map.
NAME_FONT_SIZE: int = 7


def predict_disorder(sequence: str) -> np.ndarray:
    """
    Predict the disorder score of every residue of a sequence with metapredict.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    np.ndarray
        One disorder score per residue, between 0 (ordered) and 1
        (disordered), from metapredict V3.
    """
    disorder_scores = np.asarray(
        metapredict.predict_disorder(sequence, version=METAPREDICT_VERSION), dtype=np.float64
    )

    assert disorder_scores.shape == (len(sequence),), (
        f"metapredict returned {disorder_scores.shape} scores for {len(sequence)} residues."
    )

    return disorder_scores


def residue_axis_limits(sequence_length: int) -> tuple[float, float]:
    """
    Get the limits of the residue axis, so each residue takes up one unit.

    Parameters
    ----------
    sequence_length : int
        The number of residues in the sequence.

    Returns
    -------
    tuple of float
        The lower and upper limits. Residues are at positions 1 to
        sequence_length, so the axis runs half a residue past each end.
    """
    return 0.5, sequence_length + 0.5


def draw_domain_map(axes: Axes, sequence_length: int, domains: list[ActivationDomain]) -> None:
    """
    Draw the activation domains as rectangles along a line the length of the
    sequence.

    Parameters
    ----------
    axes : Axes
        Axes the domain map is drawn on.

    sequence_length : int
        The number of residues in the sequence.

    domains : list of ActivationDomain
        The activation domains, which are drawn from half a residue before
        their first residue to half a residue after their last.
    """
    first_position, last_position = residue_axis_limits(sequence_length)
    axes.plot(
        [first_position, last_position],
        [DOMAIN_MAP_CENTER, DOMAIN_MAP_CENTER],
        color=LINE_COLOR,
        linewidth=DOMAIN_MAP_LINE_WIDTH,
        zorder=SEQUENCE_LINE_ZORDER,
    )

    for domain in domains:
        rectangle = Rectangle(
            (domain.start - 0.5, DOMAIN_MAP_CENTER - AD_RECTANGLE_HEIGHT / 2),
            width=domain.end - domain.start + 1,
            height=AD_RECTANGLE_HEIGHT,
            facecolor=AD_FACE_COLOR,
            edgecolor=LINE_COLOR,
            linewidth=AD_EDGE_LINE_WIDTH,
            zorder=AD_RECTANGLE_ZORDER,
        )
        axes.add_patch(rectangle)

    # the domain map only needs its label, so the frame and ticks are removed
    axes.set_ylim(0.0, 1.0)
    axes.set_yticks([])
    axes.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_ylabel("AD", rotation=0, horizontalalignment="right", verticalalignment="center")


def draw_score_profiles(
    axes: Axes, central_residue_scores: np.ndarray, disorder_scores: np.ndarray
) -> None:
    """
    Draw the per-residue TAD score and disorder score along the sequence.

    Parameters
    ----------
    axes : Axes
        Axes the scores are drawn on.

    central_residue_scores : np.ndarray
        Per-residue TAD scores from the 'central_residues' method, as returned
        by ``calculate_residue_scores()``. Residues no window is centered on
        are NaN, and are left out of the line.

    disorder_scores : np.ndarray
        Per-residue disorder scores, as returned by ``predict_disorder()``.
    """
    assert central_residue_scores.shape == disorder_scores.shape, (
        f"Got {central_residue_scores.shape} TAD scores but {disorder_scores.shape} "
        "disorder scores."
    )

    positions = np.arange(1, len(central_residue_scores) + 1)

    axes.plot(
        positions,
        central_residue_scores,
        color=TAD_SCORE_COLOR,
        linewidth=PROFILE_LINE_WIDTH,
        label="TAD score",
    )
    axes.plot(
        positions,
        disorder_scores,
        color=DISORDER_COLOR,
        linewidth=PROFILE_LINE_WIDTH,
        label="Disorder",
    )

    axes.set_ylim(*SCORE_AXIS_LIMITS)
    axes.set_xlabel("Residue")
    axes.set_ylabel("Score")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    # the legend sits outside the panel so it never covers the scores
    axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)


def plot_residue_profile(
    name: str,
    central_residue_scores: np.ndarray,
    disorder_scores: np.ndarray,
    domains: list[ActivationDomain],
) -> Figure:
    """
    Plot the per-residue profile of one sequence.

    Parameters
    ----------
    name : str
        Name of the sequence, written above the domain map so each page of a
        PDF of many profiles can be identified.

    central_residue_scores : np.ndarray
        Per-residue TAD scores from the 'central_residues' method, as returned
        by ``calculate_residue_scores()``.

    disorder_scores : np.ndarray
        Per-residue disorder scores, as returned by ``predict_disorder()``.

    domains : list of ActivationDomain
        The activation domains to draw on the domain map.

    Returns
    -------
    Figure
        The figure, with the domain map on top of the score panel. It is made
        without pyplot, so it is not kept open after it has been saved.
    """
    # fonts are set when text is created, so the plot settings are applied
    # while the figure is built (and again by open_profile_pdf() when saving)
    with matplotlib.rc_context(PLOT_RC_PARAMS):
        figure = Figure(figsize=FIGURE_SIZE_INCHES, layout="constrained")
        domain_map_axes, score_axes = figure.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": PANEL_HEIGHT_RATIOS}
        )

        draw_domain_map(domain_map_axes, len(central_residue_scores), domains)
        draw_score_profiles(score_axes, central_residue_scores, disorder_scores)
        score_axes.set_xlim(*residue_axis_limits(len(central_residue_scores)))

        domain_map_axes.set_title(name, loc="left", fontsize=NAME_FONT_SIZE)

    return figure


def plot_sequence_profile(
    name: str, sequence: str, window_scores: np.ndarray, domains: list[ActivationDomain]
) -> Figure:
    """
    Plot the per-residue profile of a sequence from its window scores.

    The profile always shows the per-residue TAD score from the
    'central_residues' method (whichever method the activation domains were
    called with), alongside the disorder score predicted by metapredict.

    Parameters
    ----------
    name : str
        Name of the sequence, written above the domain map.

    sequence : str
        Amino acid sequence.

    window_scores : np.ndarray
        TAD score for every 40 amino acid window of the sequence, as returned
        by ``predict_window_scores()``.

    domains : list of ActivationDomain
        The activation domains found in the sequence, drawn on the domain map.

    Returns
    -------
    Figure
        The figure (see ``plot_residue_profile()``).
    """
    central_residue_scores = calculate_residue_scores(
        sequence, window_scores, CENTRAL_RESIDUES_METHOD
    )
    disorder_scores = predict_disorder(sequence)

    return plot_residue_profile(name, central_residue_scores, disorder_scores, domains)


@contextlib.contextmanager
def open_profile_pdf(path: str) -> Iterator[PdfPages]:
    """
    Open a PDF that profiles can be added to one page at a time.

    The plot settings (Arial, and fonts that stay editable in Illustrator)
    are applied for as long as the PDF is open, because fonts are only
    embedded when it is closed.

    Parameters
    ----------
    path : str
        Path the PDF is written to.

    Yields
    ------
    PdfPages
        The open PDF. Add a figure as a new page with ``pdf.savefig(figure)``.
    """
    with matplotlib.rc_context(PLOT_RC_PARAMS), PdfPages(path) as pdf:
        yield pdf
