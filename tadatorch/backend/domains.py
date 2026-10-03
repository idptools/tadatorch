"""
Code for converting TAD scores for the windows of a sequence into annotated
activation domains.

TADA scores 40 amino acid windows and does not itself define where an
activation domain starts and stops, so the definitions used here are choices
made by tadatorch. There are two:

* 'whole_window' (the default): every residue of a window with a TAD score at
  or above the threshold is part of an activation domain. Domains are always
  at least 40 residues long, and can extend up to 39 residues past the
  residues that are responsible for the score.

* 'central_residues': only the two residues at the center of a window with a
  TAD score at or above the threshold are part of an activation domain. Gaps
  between domains that are shorter than a set size are then joined, and
  domains that are shorter than a set size are removed. This gives tighter
  domains. No window is centered on the first or last 19 residues of a
  sequence, so a domain that reaches the first (or last) residue a window can
  be centered on is extended to that end of the sequence, but only if the
  windows at that end consistently score at or above the threshold (see
  ``extend_to_termini()``).
"""

from dataclasses import dataclass

import numpy as np

from tadatorch.backend.constants import TADA_SEQUENCE_LENGTH

# Windows with a TAD score at or above this value are treated as activation
# domains. 0.5 is the threshold used to classify sequences as TADs in the TADA
# paper (Morffy, Van den Broeck et al. 2024, Nature 632, 166-173).
DEFAULT_SCORE_THRESHOLD: float = 0.5

# The two ways of deciding which residues of a window are part of an
# activation domain (see the top of this module).
WHOLE_WINDOW_METHOD: str = "whole_window"
CENTRAL_RESIDUES_METHOD: str = "central_residues"
METHODS: tuple[str, ...] = (WHOLE_WINDOW_METHOD, CENTRAL_RESIDUES_METHOD)
DEFAULT_METHOD: str = WHOLE_WINDOW_METHOD

# Number of residues at the center of a window that are marked by the
# 'central_residues' method. A window has an even number of residues, so two
# is the smallest number that sits exactly at its center.
NUMBER_OF_CENTRAL_RESIDUES: int = 2

# Defaults for the 'central_residues' method: gaps shorter than DEFAULT_GAP_SIZE
# residues are joined, and domains shorter than DEFAULT_MIN_AD_SIZE residues are
# removed. Both were picked by intuition and have not been tuned against data.
DEFAULT_GAP_SIZE: int = 6
DEFAULT_MIN_AD_SIZE: int = 6


@dataclass(frozen=True)
class ActivationDomain:
    """
    A predicted activation domain within a sequence.

    Attributes
    ----------
    start : int
        Position of the first residue of the domain. Positions are 1-indexed,
        so the first residue of the sequence is position 1.

    end : int
        Position of the last residue of the domain (1-indexed and inclusive).

    sequence : str
        Amino acid sequence of the domain.

    number_of_windows_above_threshold : int
        Number of 40 amino acid windows belonging to the domain that have a
        TAD score at or above the threshold. With the 'whole_window' method a
        window belongs to a domain if it lies within it, and with the
        'central_residues' method it belongs if its two central residues do.

    mean_score : float
        Mean TAD score of every 40 amino acid window belonging to the domain.
        This includes any windows with scores under the threshold that sit
        between windows with scores above it.

    max_score : float
        Highest TAD score of the 40 amino acid windows belonging to the
        domain.
    """

    start: int
    end: int
    sequence: str
    number_of_windows_above_threshold: int
    mean_score: float
    max_score: float


def find_contiguous_regions(is_selected: np.ndarray) -> list[tuple[int, int]]:
    """
    Find every run of consecutive True values in a boolean array.

    Parameters
    ----------
    is_selected : np.ndarray
        One-dimensional boolean array.

    Returns
    -------
    list of tuple
        A (start, end) tuple for each run of True values, in order. Indices
        are 0-indexed and end is exclusive, so a run covers
        ``is_selected[start:end]``.
    """
    # pad both ends with False so that runs touching either end of the array
    # still have a False-to-True step at their start and a True-to-False step
    # at their end
    padded = np.concatenate([[False], is_selected, [False]]).astype(int)
    steps = np.diff(padded)

    region_starts = np.flatnonzero(steps == 1)
    region_ends = np.flatnonzero(steps == -1)

    return [(int(start), int(end)) for start, end in zip(region_starts, region_ends)]


def get_marked_residues(method: str) -> tuple[int, int]:
    """
    Get which residues of a window are marked as being in an activation
    domain when the window has a score at or above the threshold.

    Parameters
    ----------
    method : str
        Either 'whole_window' or 'central_residues'.

    Returns
    -------
    int
        Position within the window of the first residue that is marked
        (0-indexed).

    int
        The number of residues that are marked.
    """
    if method == WHOLE_WINDOW_METHOD:
        return 0, TADA_SEQUENCE_LENGTH

    # the two central residues of a 40 residue window are residues 20 and 21,
    # so have 19 residues on either side of them
    first_central_residue = (TADA_SEQUENCE_LENGTH - NUMBER_OF_CENTRAL_RESIDUES) // 2
    return first_central_residue, NUMBER_OF_CENTRAL_RESIDUES


def mark_residues(
    is_above_threshold: np.ndarray, sequence_length: int, first_marked: int, number_marked: int
) -> np.ndarray:
    """
    Mark the residues that are in an activation domain.

    Parameters
    ----------
    is_above_threshold : np.ndarray
        Whether each window has a TAD score at or above the threshold.

    sequence_length : int
        The number of residues in the sequence.

    first_marked : int
        Position within a window of the first residue it marks (0-indexed).

    number_marked : int
        The number of residues each window marks.

    Returns
    -------
    np.ndarray
        Boolean array with one value per residue, which is True for residues
        marked by at least one window at or above the threshold.
    """
    is_in_domain = np.zeros(sequence_length, dtype=bool)

    for window_start in np.flatnonzero(is_above_threshold):
        first_residue = window_start + first_marked
        is_in_domain[first_residue : first_residue + number_marked] = True

    return is_in_domain


def join_short_gaps(is_in_domain: np.ndarray, gap_size: int) -> np.ndarray:
    """
    Join activation domains that are separated by a short gap.

    Parameters
    ----------
    is_in_domain : np.ndarray
        Boolean array with one value per residue, which is True for residues
        in an activation domain.

    gap_size : int
        Gaps between two domains that are shorter than this many residues are
        made part of the domain.

    Returns
    -------
    np.ndarray
        A copy of ``is_in_domain`` in which the residues of every short gap
        are True. The residues before the first domain and after the last
        domain are not gaps, so are never changed.
    """
    joined = is_in_domain.copy()

    for gap_start, gap_end in find_contiguous_regions(~is_in_domain):
        is_between_domains = gap_start > 0 and gap_end < len(is_in_domain)
        if is_between_domains and gap_end - gap_start < gap_size:
            joined[gap_start:gap_end] = True

    return joined


def remove_short_domains(is_in_domain: np.ndarray, min_ad_size: int) -> np.ndarray:
    """
    Remove activation domains that are too short.

    Parameters
    ----------
    is_in_domain : np.ndarray
        Boolean array with one value per residue, which is True for residues
        in an activation domain.

    min_ad_size : int
        Domains shorter than this many residues are removed.

    Returns
    -------
    np.ndarray
        A copy of ``is_in_domain`` in which the residues of every domain that
        is too short are False.
    """
    kept = is_in_domain.copy()

    for domain_start, domain_end in find_contiguous_regions(is_in_domain):
        if domain_end - domain_start < min_ad_size:
            kept[domain_start:domain_end] = False

    return kept


def extend_to_termini(
    is_in_domain: np.ndarray,
    is_marked: np.ndarray,
    first_markable: int,
    last_markable: int,
    min_ad_size: int,
) -> np.ndarray:
    """
    Extend activation domains at the ends of a sequence out to its termini.

    With the 'central_residues' method no window is centered on the residues
    at either end of a sequence, so they can never be marked. To stop domains
    at the ends of a sequence being cut short, the residues before the first
    markable residue are added to the domain that starts there, and the
    residues after the last markable residue are added to the domain that
    ends there.

    An end is only extended if a run of ``min_ad_size`` residues starting at
    the first (or ending at the last) markable residue were all marked
    directly by windows at or above the threshold. This means the windows at
    that end must consistently score at or above the threshold: a single
    window at the very end of a sequence is not enough, and neither are
    residues that are only part of a domain because a gap was joined.
    Without this, one window would be enough to call a domain of around 20
    residues at a terminus, which would make domains much easier to call at
    the ends of a sequence than in the middle of it.

    Parameters
    ----------
    is_in_domain : np.ndarray
        Boolean array with one value per residue, which is True for residues
        in an activation domain (after gaps have been joined and short
        domains removed).

    is_marked : np.ndarray
        Boolean array with one value per residue, which is True for residues
        marked directly by a window at or above the threshold.

    first_markable : int
        Position of the first residue a window can mark (0-indexed).

    last_markable : int
        Position of the last residue a window can mark (0-indexed).

    min_ad_size : int
        The number of residues at an end that must all be marked directly for
        that end to be extended.

    Returns
    -------
    np.ndarray
        A copy of ``is_in_domain`` in which the residues beyond the markable
        residues are True at each end that is extended.
    """
    extended = is_in_domain.copy()

    # at least one residue must be marked, or an end with no domain at all
    # would be extended when min_ad_size is 0
    number_required = max(min_ad_size, 1)

    # a sequence with fewer markable residues than are required cannot have
    # either end extended
    number_markable = last_markable - first_markable + 1
    if number_required > number_markable:
        return extended

    if is_marked[first_markable : first_markable + number_required].all():
        extended[:first_markable] = True

    if is_marked[last_markable - number_required + 1 : last_markable + 1].all():
        extended[last_markable + 1 :] = True

    return extended


def find_activation_domains(
    sequence: str,
    window_scores: list[float] | np.ndarray,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    method: str = DEFAULT_METHOD,
    gap_size: int = DEFAULT_GAP_SIZE,
    min_ad_size: int = DEFAULT_MIN_AD_SIZE,
) -> list[ActivationDomain]:
    """
    Annotate the activation domains in a sequence from its window scores.

    With the default 'whole_window' method, every 40 amino acid window with a
    TAD score at or above the threshold marks all 40 of its residues as being
    part of an activation domain. Each contiguous stretch of marked residues
    is then one activation domain, so windows that overlap (or directly
    follow one another) end up in the same domain and domains are always at
    least 40 residues long.

    With the 'central_residues' method, each window at or above the threshold
    only marks its two central residues. Gaps between stretches of marked
    residues that are shorter than ``gap_size`` are then joined, after which
    stretches shorter than ``min_ad_size`` are removed. Finally, domains at
    either end of the sequence are extended to the terminus if the windows
    at that end consistently reach the threshold (see ``extend_to_termini()``).
    What is left are the activation domains.

    Parameters
    ----------
    sequence : str
        Amino acid sequence at least 40 residues long.

    window_scores : list of float or np.ndarray
        TAD score for every 40 amino acid window of the sequence, in order,
        where the window at index i covers ``sequence[i:i + 40]``. There must
        be len(sequence) - 39 scores.

    threshold : float
        Windows with a TAD score at or above this value (between 0 and 1) are
        treated as activation domains. Default is 0.5.

    method : str
        Which residues of a window are marked: 'whole_window' or
        'central_residues'. Default is 'whole_window'.

    gap_size : int
        Gaps between domains shorter than this many residues are joined. Only
        used by the 'central_residues' method. Default is 6.

    min_ad_size : int
        Domains shorter than this many residues are removed (after gaps have
        been joined). This is also the number of residues at an end of the
        sequence that must be marked for a domain to be extended to that
        terminus. Only used by the 'central_residues' method. Default is 6.

    Returns
    -------
    list of ActivationDomain
        The activation domains in the sequence, in order from the N- to the
        C-terminus. The list is empty if there are none.

    Raises
    ------
    ValueError
        If the threshold is not between 0 and 1, the method is not one of the
        two options, gap_size or min_ad_size are negative, or the number of
        scores does not match the number of windows in the sequence.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"The threshold must be between 0 and 1, but got {threshold}.")
    if method not in METHODS:
        raise ValueError(f"The method must be one of {METHODS}, but got '{method}'.")
    if gap_size < 0 or min_ad_size < 0:
        raise ValueError("gap_size and min_ad_size cannot be negative.")

    scores = np.asarray(window_scores, dtype=np.float64)

    number_of_windows = len(sequence) - TADA_SEQUENCE_LENGTH + 1
    if scores.shape != (number_of_windows,):
        raise ValueError(
            f"A sequence of {len(sequence)} residues has {number_of_windows} windows of "
            f"{TADA_SEQUENCE_LENGTH}, but got scores with shape {scores.shape}."
        )

    is_above_threshold = scores >= threshold

    first_marked, number_marked = get_marked_residues(method)
    is_marked = mark_residues(is_above_threshold, len(sequence), first_marked, number_marked)
    is_in_domain = is_marked

    if method == CENTRAL_RESIDUES_METHOD:
        is_in_domain = join_short_gaps(is_in_domain, gap_size)
        is_in_domain = remove_short_domains(is_in_domain, min_ad_size)

        # the first window marks the first markable residue, and the last
        # window marks the last markable residue
        last_markable = (number_of_windows - 1) + first_marked + number_marked - 1
        is_in_domain = extend_to_termini(
            is_in_domain, is_marked, first_marked, last_markable, min_ad_size
        )

    domains = []
    for region_start, region_end in find_contiguous_regions(is_in_domain):
        # the windows belonging to this region are those whose marked residues
        # are all within it: the first one marks the first residue of the
        # region, and the last one marks its last residue. A region that has
        # been extended to a terminus runs past the residues any window marks,
        # so starts with the first window (or ends with the last window).
        first_window = max(region_start - first_marked, 0)
        last_window = min(region_end - first_marked - number_marked, number_of_windows - 1)
        scores_in_region = scores[first_window : last_window + 1]
        windows_above_threshold = is_above_threshold[first_window : last_window + 1]

        domains.append(
            ActivationDomain(
                # region_start is 0-indexed, and region_end is 0-indexed and
                # exclusive (which is the same number as 1-indexed and inclusive)
                start=region_start + 1,
                end=region_end,
                sequence=sequence[region_start:region_end],
                number_of_windows_above_threshold=int(windows_above_threshold.sum()),
                mean_score=float(scores_in_region.mean()),
                max_score=float(scores_in_region.max()),
            )
        )

    return domains
