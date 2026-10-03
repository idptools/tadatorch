"""
Tests for converting window scores into annotated activation domains.
"""

import numpy as np
import pytest

from tadatorch.backend.constants import TADA_SEQUENCE_LENGTH
from tadatorch.backend.domains import (
    CENTRAL_RESIDUES_METHOD,
    find_activation_domains,
    find_contiguous_regions,
    join_short_gaps,
    remove_short_domains,
)
from tadatorch.tada import predict_activation_domains
from tadatorch.tests.reference_scores import TADATORCH_SCORES

# Scores used to build window scores by hand, on either side of the default
# threshold of 0.5.
LOW_SCORE: float = 0.1
HIGH_SCORE: float = 0.9

EXACT_SEQUENCE: str = "EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD"

# Largest difference allowed between scores that should be the same.
SCORE_TOLERANCE: float = 0.000001


def make_test_sequence(length: int) -> str:
    """
    Make a sequence where every position can be told apart when slicing.

    Parameters
    ----------
    length : int
        Number of residues in the sequence.

    Returns
    -------
    str
        Sequence of the requested length that cycles through the amino acids.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(amino_acids[index % len(amino_acids)] for index in range(length))


def make_window_scores(sequence: str, high_scoring_windows: list[int]) -> np.ndarray:
    """
    Make window scores for a sequence where only chosen windows score highly.

    Parameters
    ----------
    sequence : str
        Sequence the scores are for.

    high_scoring_windows : list of int
        Indices (0-indexed) of the windows given HIGH_SCORE. Every other
        window is given LOW_SCORE.

    Returns
    -------
    np.ndarray
        One score for every 40 amino acid window of the sequence.
    """
    number_of_windows = len(sequence) - TADA_SEQUENCE_LENGTH + 1
    window_scores = np.full(number_of_windows, LOW_SCORE)
    window_scores[high_scoring_windows] = HIGH_SCORE
    return window_scores


def test_find_contiguous_regions() -> None:
    """Runs of True should be found anywhere in the array, including both ends."""
    is_selected = np.array([True, True, False, True, False, False, True], dtype=bool)
    assert find_contiguous_regions(is_selected) == [(0, 2), (3, 4), (6, 7)]

    assert find_contiguous_regions(np.zeros(5, dtype=bool)) == []
    assert find_contiguous_regions(np.ones(5, dtype=bool)) == [(0, 5)]


def test_no_domains_when_all_scores_are_low() -> None:
    """A sequence with no window at the threshold should have no domains."""
    sequence = make_test_sequence(100)
    assert find_activation_domains(sequence, make_window_scores(sequence, [])) == []


def test_single_window_gives_a_40_residue_domain() -> None:
    """One high-scoring window should give a domain that is exactly that window."""
    sequence = make_test_sequence(100)
    domains = find_activation_domains(sequence, make_window_scores(sequence, [10]))

    assert len(domains) == 1
    domain = domains[0]

    # window 10 (0-indexed) covers residues 11 to 50 (1-indexed, inclusive)
    assert (domain.start, domain.end) == (11, 50)
    assert domain.sequence == sequence[10:50]
    assert domain.number_of_windows_above_threshold == 1
    assert domain.mean_score == pytest.approx(HIGH_SCORE)
    assert domain.max_score == pytest.approx(HIGH_SCORE)


def test_overlapping_windows_are_merged() -> None:
    """
    High-scoring windows whose residues overlap should end up in one domain,
    even if the windows between them score below the threshold.
    """
    sequence = make_test_sequence(100)
    domains = find_activation_domains(sequence, make_window_scores(sequence, [10, 11, 20]))

    assert len(domains) == 1
    domain = domains[0]

    # windows 10 to 20 cover residues 11 to 60
    assert (domain.start, domain.end) == (11, 60)
    assert domain.sequence == sequence[10:60]
    assert domain.number_of_windows_above_threshold == 3

    # 11 windows lie within the domain: 3 high-scoring and 8 low-scoring
    expected_mean = (3 * HIGH_SCORE + 8 * LOW_SCORE) / 11
    assert domain.mean_score == pytest.approx(expected_mean)
    assert domain.max_score == pytest.approx(HIGH_SCORE)


def test_separate_windows_give_separate_domains() -> None:
    """High-scoring windows that share no residues should give separate domains."""
    sequence = make_test_sequence(100)
    domains = find_activation_domains(sequence, make_window_scores(sequence, [0, 60]))

    # windows 0 and 60 cover residues 1 to 40 and 61 to 100, with a gap between
    assert [(domain.start, domain.end) for domain in domains] == [(1, 40), (61, 100)]


def test_adjacent_windows_give_one_domain() -> None:
    """Windows that directly follow one another leave no gap, so are one domain."""
    sequence = make_test_sequence(100)
    domains = find_activation_domains(sequence, make_window_scores(sequence, [0, 40]))

    assert [(domain.start, domain.end) for domain in domains] == [(1, 80)]


def test_threshold_is_inclusive_and_adjustable() -> None:
    """Windows exactly at the threshold count, and the threshold can be changed."""
    sequence = make_test_sequence(50)
    window_scores = make_window_scores(sequence, [0])

    assert len(find_activation_domains(sequence, window_scores, threshold=HIGH_SCORE)) == 1
    assert find_activation_domains(sequence, window_scores, threshold=0.95) == []

    # with a threshold under the low score every window counts, so the whole
    # sequence is one domain
    domains = find_activation_domains(sequence, window_scores, threshold=0.05)
    assert [(domain.start, domain.end) for domain in domains] == [(1, 50)]


def test_find_activation_domains_rejects_bad_input() -> None:
    """The wrong number of scores, or an impossible threshold, should fail."""
    sequence = make_test_sequence(100)
    window_scores = make_window_scores(sequence, [])

    with pytest.raises(ValueError):
        find_activation_domains(sequence, window_scores[:-1])
    with pytest.raises(ValueError):
        find_activation_domains(sequence, window_scores, threshold=1.5)
    with pytest.raises(ValueError):
        find_activation_domains(sequence, window_scores, threshold=-0.1)


def test_join_short_gaps() -> None:
    """Only gaps between domains that are shorter than the gap size should be filled."""
    #                       0      1      2      3     4      5      6     7      8
    is_in_domain = np.array([False, True, False, False, True, False, False, False, True])

    # the gap of 2 is joined but the gap of 3 is not, and the leading gap is left alone
    joined = join_short_gaps(is_in_domain, gap_size=3)
    assert list(joined) == [False, True, True, True, True, False, False, False, True]

    assert list(join_short_gaps(is_in_domain, gap_size=4)) == [False] + [True] * 8
    assert list(join_short_gaps(is_in_domain, gap_size=0)) == list(is_in_domain)


def test_remove_short_domains() -> None:
    """Domains shorter than the minimum size should be removed."""
    is_in_domain = np.array([True, False, True, True, False, True, True, True])

    assert list(remove_short_domains(is_in_domain, min_ad_size=3)) == [False] * 5 + [True] * 3
    assert list(remove_short_domains(is_in_domain, min_ad_size=1)) == list(is_in_domain)


def test_central_residues_marks_the_middle_of_each_window() -> None:
    """
    With the central residues method a window should only mark its two
    central residues, and a run of windows should mark one more residue than
    there are windows.
    """
    sequence = make_test_sequence(100)

    # window 10 (0-indexed) covers residues 11 to 50, so its center is 30 and 31
    one_window = find_activation_domains(
        sequence, make_window_scores(sequence, [10]), method=CENTRAL_RESIDUES_METHOD, min_ad_size=2
    )
    assert [(domain.start, domain.end) for domain in one_window] == [(30, 31)]
    assert one_window[0].sequence == sequence[29:31]

    # five windows in a row mark six residues, which is just long enough to keep
    five_windows = find_activation_domains(
        sequence, make_window_scores(sequence, [10, 11, 12, 13, 14]), method=CENTRAL_RESIDUES_METHOD
    )
    assert [(domain.start, domain.end) for domain in five_windows] == [(30, 35)]
    assert five_windows[0].number_of_windows_above_threshold == 5
    assert five_windows[0].mean_score == pytest.approx(HIGH_SCORE)


def test_central_residues_removes_short_domains() -> None:
    """A single window gives two residues, which is under the default minimum of six."""
    sequence = make_test_sequence(100)
    window_scores = make_window_scores(sequence, [10])

    assert find_activation_domains(sequence, window_scores, method=CENTRAL_RESIDUES_METHOD) == []


def test_central_residues_joins_short_gaps() -> None:
    """
    Two runs of windows separated by a gap of four residues should be joined
    with the default gap size of six, but not with a gap size of four.
    """
    sequence = make_test_sequence(100)

    # windows 10 to 14 mark residues 30 to 35, and windows 20 to 24 mark
    # residues 40 to 45, which leaves residues 36 to 39 as a gap of four
    window_scores = make_window_scores(sequence, [10, 11, 12, 13, 14, 20, 21, 22, 23, 24])

    joined = find_activation_domains(sequence, window_scores, method=CENTRAL_RESIDUES_METHOD)
    assert [(domain.start, domain.end) for domain in joined] == [(30, 45)]

    # the domain now includes the five low-scoring windows centered on the gap
    assert joined[0].number_of_windows_above_threshold == 10
    assert joined[0].mean_score == pytest.approx((10 * HIGH_SCORE + 5 * LOW_SCORE) / 15)
    assert joined[0].max_score == pytest.approx(HIGH_SCORE)

    separate = find_activation_domains(
        sequence, window_scores, method=CENTRAL_RESIDUES_METHOD, gap_size=4
    )
    assert [(domain.start, domain.end) for domain in separate] == [(30, 35), (40, 45)]


def test_central_residues_extends_to_the_termini() -> None:
    """
    No window is centered on the first or last 19 residues, so a domain should
    be extended to an end of the sequence when the windows at that end are
    consistently above the threshold.
    """
    sequence = make_test_sequence(100)
    last_window = len(sequence) - TADA_SEQUENCE_LENGTH

    # the first five windows mark residues 20 to 25, which is six residues
    # starting at the first residue a window can mark
    n_terminal = find_activation_domains(
        sequence, make_window_scores(sequence, [0, 1, 2, 3, 4]), method=CENTRAL_RESIDUES_METHOD
    )
    assert [(domain.start, domain.end) for domain in n_terminal] == [(1, 25)]
    assert n_terminal[0].sequence == sequence[:25]
    assert n_terminal[0].number_of_windows_above_threshold == 5
    assert n_terminal[0].mean_score == pytest.approx(HIGH_SCORE)

    # the last five windows mark residues 76 to 81, where 81 is the last
    # residue a window can mark
    last_five_windows = list(range(last_window - 4, last_window + 1))
    c_terminal = find_activation_domains(
        sequence, make_window_scores(sequence, last_five_windows), method=CENTRAL_RESIDUES_METHOD
    )
    assert [(domain.start, domain.end) for domain in c_terminal] == [(76, 100)]
    assert c_terminal[0].number_of_windows_above_threshold == 5

    # with every window above the threshold the whole sequence is one domain
    every_window = list(range(last_window + 1))
    whole_sequence = find_activation_domains(
        sequence, make_window_scores(sequence, every_window), method=CENTRAL_RESIDUES_METHOD
    )
    assert [(domain.start, domain.end) for domain in whole_sequence] == [(1, 100)]
    assert whole_sequence[0].number_of_windows_above_threshold == len(every_window)


def test_one_window_at_a_terminus_is_not_extended() -> None:
    """
    A terminus should not be extended on the strength of too few windows:
    one window at the very start of a sequence should give no domain at all,
    and four windows (five residues) are still under the default of six.
    """
    sequence = make_test_sequence(100)

    for windows in ([0], [0, 1, 2, 3]):
        domains = find_activation_domains(
            sequence, make_window_scores(sequence, windows), method=CENTRAL_RESIDUES_METHOD
        )
        assert domains == []


def test_joined_gap_at_a_terminus_is_not_extended() -> None:
    """
    The residues at a terminus have to be marked directly by windows for it
    to be extended. Here the first window is above the threshold but the next
    two are not, and the domain only reaches the first markable residue
    because the gap they leave is joined, so it should not be extended.
    """
    sequence = make_test_sequence(100)

    # window 0 marks residues 20 and 21, and windows 3 to 7 mark residues 23
    # to 28, which leaves residue 22 as a gap that is joined
    domains = find_activation_domains(
        sequence, make_window_scores(sequence, [0, 3, 4, 5, 6, 7]), method=CENTRAL_RESIDUES_METHOD
    )
    assert [(domain.start, domain.end) for domain in domains] == [(20, 28)]


def test_domain_in_the_middle_is_not_extended() -> None:
    """A domain that does not reach the first or last markable residue is left as it is."""
    sequence = make_test_sequence(100)

    domains = find_activation_domains(
        sequence, make_window_scores(sequence, [1, 2, 3, 4, 5]), method=CENTRAL_RESIDUES_METHOD
    )
    assert [(domain.start, domain.end) for domain in domains] == [(21, 26)]


def test_extension_needs_at_least_one_window() -> None:
    """With a minimum size of 0 a terminus with no domain should still not be extended."""
    sequence = make_test_sequence(100)

    domains = find_activation_domains(
        sequence, make_window_scores(sequence, [30]), method=CENTRAL_RESIDUES_METHOD, min_ad_size=0
    )
    assert [(domain.start, domain.end) for domain in domains] == [(50, 51)]


def test_sequence_too_short_to_extend() -> None:
    """
    A 40 residue sequence has one window and so two markable residues, which
    is fewer than the default minimum size of six, so it has no domain. With
    a minimum size of two the one window is enough to extend both ends.
    """
    sequence = make_test_sequence(TADA_SEQUENCE_LENGTH)
    window_scores = make_window_scores(sequence, [0])

    assert find_activation_domains(sequence, window_scores, method=CENTRAL_RESIDUES_METHOD) == []

    domains = find_activation_domains(
        sequence, window_scores, method=CENTRAL_RESIDUES_METHOD, min_ad_size=2
    )
    assert [(domain.start, domain.end) for domain in domains] == [(1, TADA_SEQUENCE_LENGTH)]


def test_find_activation_domains_rejects_bad_settings() -> None:
    """A method that does not exist, or negative sizes, should fail."""
    sequence = make_test_sequence(100)
    window_scores = make_window_scores(sequence, [10])

    with pytest.raises(ValueError):
        find_activation_domains(sequence, window_scores, method="middle")
    with pytest.raises(ValueError):
        find_activation_domains(sequence, window_scores, gap_size=-1)
    with pytest.raises(ValueError):
        find_activation_domains(sequence, window_scores, min_ad_size=-1)


def test_predict_activation_domains() -> None:
    """
    A 40 amino acid sequence has one window, so above the threshold it is one
    domain covering the whole sequence with that window's score.
    """
    domains = predict_activation_domains(EXACT_SEQUENCE)

    assert len(domains) == 1
    domain = domains[0]

    assert (domain.start, domain.end) == (1, TADA_SEQUENCE_LENGTH)
    assert domain.sequence == EXACT_SEQUENCE
    assert domain.max_score == pytest.approx(TADATORCH_SCORES[EXACT_SEQUENCE], abs=SCORE_TOLERANCE)

    # the score of this sequence is about 0.648, so a higher threshold removes it
    assert predict_activation_domains(EXACT_SEQUENCE, threshold=0.7) == []


def test_predict_activation_domains_rejects_bad_sequences() -> None:
    """Sequences that are too short or have invalid residues should fail."""
    with pytest.raises(ValueError):
        predict_activation_domains(EXACT_SEQUENCE[:-1])
    with pytest.raises(ValueError):
        predict_activation_domains(EXACT_SEQUENCE.lower())
