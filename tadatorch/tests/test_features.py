"""
Tests for the feature calculation and scaling.
"""

import numpy as np
import pytest
from sparrow import Protein

from tadatorch.backend.constants import (
    AMINO_ACIDS,
    FEATURE_WINDOW_SIZE,
    NUMBER_OF_FEATURE_WINDOWS,
    NUMBER_OF_FEATURES,
)
from tadatorch.backend.features import (
    PLDDT_TABLE_UNIT,
    RESIDUE_GROUPS,
    calculate_kappa,
    calculate_omega,
    calculate_window_features,
    create_features,
    load_plddt_table,
    load_scaler_metrics,
    number_window,
    scale_features,
    split_into_feature_windows,
)

TEST_SEQUENCE: str = "EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD"

# Positions of features in the 42 features calculated for every window.
KAPPA_INDEX: int = 0
OMEGA_INDEX: int = 1
FIRST_GROUP_COUNT_INDEX: int = 10
PLDDT_INDEX: int = 21
FIRST_AMINO_ACID_COUNT_INDEX: int = 22

# Number of features that are specific to a window (all but kappa and Omega).
NUMBER_OF_WINDOW_FEATURES: int = NUMBER_OF_FEATURES - 2


def test_split_into_feature_windows() -> None:
    """A 40 amino acid sequence should give 36 windows of 5 that step by 1."""
    windows = split_into_feature_windows(TEST_SEQUENCE)

    assert len(windows) == NUMBER_OF_FEATURE_WINDOWS
    assert windows[0] == "EFSPE"
    assert windows[1] == "FSPEN"
    assert windows[-1] == "SSPTD"


def test_calculate_window_features_counts() -> None:
    """
    Check the counting features for a window where the right answers can be
    worked out by hand.
    """
    window_features = calculate_window_features("EFSPE")
    assert len(window_features) == NUMBER_OF_WINDOW_FEATURES

    # window features do not include kappa and Omega, so positions are shifted by 2
    first_group_count = FIRST_GROUP_COUNT_INDEX - 2
    group_counts = window_features[first_group_count : first_group_count + len(RESIDUE_GROUPS)]
    # aliphatic, aromatic, branching, charged, negative, phosphorylatable,
    # polar, hydrophobic, positive, sulfur containing, tiny
    assert group_counts == (0, 1, 0, 2, 2, 1, 2, 1, 0, 0, 2)

    amino_acid_counts = window_features[FIRST_AMINO_ACID_COUNT_INDEX - 2 :]
    expected_counts = {"E": 2, "F": 1, "S": 1, "P": 1}
    assert amino_acid_counts == tuple(
        expected_counts.get(amino_acid, 0) for amino_acid in AMINO_ACIDS
    )


def test_calculate_window_features_properties() -> None:
    """
    Check the physicochemical features for a window where the right answers
    can be worked out by hand.
    """
    hydropathy, wimley_white, ncpr, disorder_promoting, fcr, net_charge, negative, positive = (
        calculate_window_features("EFSPE")[:8]
    )

    # Kyte-Doolittle values shifted to start at 0: E 1.0, F 7.3, S 3.7, P 2.9
    assert hydropathy == pytest.approx((1.0 + 7.3 + 3.7 + 2.9 + 1.0) / 5)
    # Wimley-White values: E -2.02, F 1.13, S -0.13, P -0.45
    assert wimley_white == pytest.approx((-2.02 + 1.13 - 0.13 - 0.45 - 2.02) / 5)

    # two negative residues (E) and no positive residues out of five
    assert ncpr == pytest.approx(-0.4)
    assert fcr == pytest.approx(0.4)
    assert net_charge == pytest.approx(0.4)
    assert negative == pytest.approx(0.4)
    assert positive == 0.0

    # every residue apart from F is disorder promoting
    assert disorder_promoting == pytest.approx(0.8)


def test_number_window() -> None:
    """Every window should get its own number, counting in base 20."""
    first_amino_acid, second_amino_acid, last_amino_acid = (
        AMINO_ACIDS[0],
        AMINO_ACIDS[1],
        AMINO_ACIDS[-1],
    )

    assert number_window(first_amino_acid * 5) == 0
    assert number_window(first_amino_acid * 4 + second_amino_acid) == 1
    assert number_window(second_amino_acid + first_amino_acid * 4) == 20**4
    assert number_window(last_amino_acid * 5) == 20**5 - 1


def test_plddt_table() -> None:
    """The table should have a sensible mean pLDDT score for every window."""
    table = load_plddt_table()
    assert table.shape == (20**5,)

    # pLDDT scores are between 0 and 100
    mean_scores = table * PLDDT_TABLE_UNIT / FEATURE_WINDOW_SIZE
    assert mean_scores.min() >= 0.0
    assert mean_scores.max() <= 100.0


def test_plddt_table_matches_alphapredict() -> None:
    """
    The table was built with alphaPredict, so if alphaPredict is installed
    make sure looking a window up gives what alphaPredict predicts for it.
    """
    alphapredict = pytest.importorskip("alphaPredict")

    for window in ("EFSPE", "GSGSG", "WWLLF", "KRKRK", "ACDEF"):
        expected = sum(alphapredict.predict(window)) / len(window)
        looked_up = calculate_window_features(window)[PLDDT_INDEX - 2]
        assert looked_up == pytest.approx(expected, abs=1e-9)


def test_kappa_separates_mixed_and_segregated_charges() -> None:
    """Kappa should be low when charges are well mixed and 1 when they are segregated."""
    well_mixed = "EK" * 20
    segregated = "E" * 20 + "K" * 20

    kappa = calculate_kappa([well_mixed, segregated])

    assert 0.0 <= kappa[0] < 0.1
    assert kappa[1] == pytest.approx(1.0)


def test_kappa_and_omega_match_sparrow() -> None:
    """Kappa should be what sparrow reports for a protein, for every sequence in a batch."""
    sequences = [TEST_SEQUENCE, TEST_SEQUENCE[::-1], "EKDR" * 10]
    kappa = calculate_kappa(sequences)

    for sequence, value in zip(sequences, kappa):
        assert value == pytest.approx(Protein(sequence).kappa, abs=1e-12)

    omega = calculate_omega(sequences)
    assert np.all((omega >= 0.0) & (omega <= 1.0))


def test_kappa_and_omega_are_undefined_without_the_groups_compared() -> None:
    """
    Kappa should be -1 without both positive and negative residues, and Omega
    should be -1 without any charged or proline residues.
    """
    no_charges = "GS" * 20
    only_positive = "GSGSK" * 8
    assert list(calculate_kappa([no_charges, only_positive])) == [-1.0, -1.0]

    no_charged_or_proline = "QNGSTA" * 6 + "QNGS"
    assert list(calculate_omega([no_charged_or_proline])) == [-1.0]


def test_create_features_shape_and_structure() -> None:
    """Check the shape of the features and properties that must always hold."""
    features = create_features([TEST_SEQUENCE, TEST_SEQUENCE[::-1]])

    assert features.shape == (2, NUMBER_OF_FEATURE_WINDOWS, NUMBER_OF_FEATURES)
    assert np.all(np.isfinite(features))

    # kappa and Omega are whole-sequence values, so are the same for every window
    for index in (KAPPA_INDEX, OMEGA_INDEX):
        assert np.all(features[0, :, index] == features[0, 0, index])

    # the 20 amino acid counts must add up to the size of the window
    amino_acid_counts = features[:, :, FIRST_AMINO_ACID_COUNT_INDEX:]
    assert np.all(amino_acid_counts.sum(axis=2) == FEATURE_WINDOW_SIZE)


def test_create_features_rejects_bad_sequences() -> None:
    """Sequences that are the wrong length or have invalid residues should fail."""
    with pytest.raises(ValueError):
        create_features([])
    with pytest.raises(ValueError):
        create_features([TEST_SEQUENCE[:-1]])
    with pytest.raises(ValueError):
        create_features([TEST_SEQUENCE + "A"])
    with pytest.raises(ValueError):
        create_features([TEST_SEQUENCE.lower()])
    with pytest.raises(ValueError):
        create_features(["X" + TEST_SEQUENCE[1:]])


def test_scale_features_matches_formula() -> None:
    """
    Scaling should standardize each feature and then min-max scale the
    result, using the metrics for that feature.
    """
    scaler_metrics = load_scaler_metrics()
    features = create_features([TEST_SEQUENCE])
    scaled_features = scale_features(features, scaler_metrics)

    assert scaled_features.shape == features.shape

    # check every feature one at a time against the scaler columns written out
    # by position: mean (0), standard deviation (2), minimum (5) and range (9)
    for feature_index in range(NUMBER_OF_FEATURES):
        mean, _, standard_deviation, _, _, data_min, _, _, _, data_range = scaler_metrics[
            feature_index
        ]
        standardized = (features[0, :, feature_index] - mean) / standard_deviation
        expected = (standardized - data_min) / data_range
        np.testing.assert_array_equal(scaled_features[0, :, feature_index], expected)


def test_scale_features_does_not_modify_input() -> None:
    """Scaling should return a new array rather than changing the input."""
    features = create_features([TEST_SEQUENCE])
    features_before = features.copy()

    scale_features(features, load_scaler_metrics())

    np.testing.assert_array_equal(features, features_before)
