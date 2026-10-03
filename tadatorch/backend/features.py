"""
Code for calculating and scaling the sequence features TADA makes predictions
from.

The features (and the order they are in) are fixed by how the original TADA
network was trained (https://github.com/LisaVdB/TADA). Sequence parameters
are calculated with sparrow (https://github.com/idptools/sparrow).

Two of the 42 features (kappa and Omega) are not numerically identical to
those TADA was trained with, which were calculated by localcider. sparrow
uses a slightly different (and much faster) definition of kappa, and returns
-1 (undefined) for kappa if a sequence does not have both positive and
negative residues, whereas localcider only did so if it had neither.
See the README for how much this changes TAD scores.
"""

import functools
import importlib.resources

import numpy as np
from sparrow.data import amino_acids as sparrow_amino_acids
from sparrow.patterning import vectorized as sparrow_patterning
from sparrow.sequence_analysis import physical_properties

from tadatorch.backend.constants import (
    AMINO_ACIDS,
    FEATURE_WINDOW_SIZE,
    FEATURE_WINDOW_STEP,
    NUMBER_OF_FEATURE_WINDOWS,
    NUMBER_OF_FEATURES,
    TADA_SEQUENCE_LENGTH,
)
from tadatorch.backend.utils import validate_sequence

# Groups of residues that are counted in every feature window. The order of the
# groups is the order of the corresponding features, so must not be changed.
RESIDUE_GROUPS: dict[str, frozenset[str]] = {
    "aliphatic": frozenset("IVLA"),
    "aromatic": frozenset("WFY"),
    "branching": frozenset("VIT"),
    "charged": frozenset("KRHDE"),
    "negative": frozenset("DE"),
    "phosphorylatable": frozenset("STY"),
    "polar": frozenset("RKDEQNY"),
    "hydrophobic": frozenset("WFLVICM"),
    "positive": frozenset("KRH"),
    "sulfur_containing": frozenset("MC"),
    "tiny": frozenset("GASP"),
}

# Wimley-White whole-residue interface hydrophobicity scale, in kcal/mol: the
# free energy of transfer from a bilayer interface to water, so that positive
# values are hydrophobic (Wimley & White 1996, Nat. Struct. Biol. 3, 842-848).
# His, Lys, Arg, Asp and Glu use the values for their charged forms. These are
# the values TADA was trained with (they are what localcider uses), and differ
# for His, Lys and Arg from the version of the scale in AAindex.
WIMLEY_WHITE_SCALE: dict[str, float] = {
    "A": -0.17, "C": 0.24, "D": -1.23, "E": -2.02, "F": 1.13,
    "G": -0.01, "H": -0.96, "I": 0.31, "K": -0.99, "L": 0.56,
    "M": 0.23, "N": -0.42, "P": -0.45, "Q": -0.58, "R": -0.81,
    "S": -0.13, "T": -0.14, "V": -0.07, "W": 1.85, "Y": 0.94,
}  # fmt: skip

# Residue groups whose patterning kappa and Omega describe. Kappa is the
# patterning of positive residues with respect to negative residues (Das &
# Pappu 2013, Proc. Natl. Acad. Sci. 110, 13392-13397). Omega is the patterning
# of charged and proline residues with respect to all other residues (Martin et
# al. 2016, J. Am. Chem. Soc. 138, 15323-15335).
KAPPA_POSITIVE_RESIDUES: list[str] = ["R", "K"]
KAPPA_NEGATIVE_RESIDUES: list[str] = ["E", "D"]
OMEGA_RESIDUES: list[str] = ["P", "E", "D", "K", "R"]

# Kappa is the average of the values calculated with these two blob sizes,
# which is how kappa is defined in both localcider and sparrow.
KAPPA_BLOB_SIZES: tuple[int, int] = (5, 6)

# Number of feature windows whose features are kept in memory so they are not
# re-calculated. When TADA is slid along a protein, each 40 amino acid sequence
# shares 35 of its 36 feature windows with the one before it, so re-using them
# saves most of the work. Each entry is 40 numbers, so this is around 50 MB.
WINDOW_FEATURE_CACHE_SIZE: int = 65536

# The table of predicted pLDDT scores holds, for every possible 5-residue
# window, the sum of the five per-residue scores from alphaPredict in units of
# this size (alphaPredict rounds its scores to four decimal places).
PLDDT_TABLE_FILE: str = "five_mer_plddt_sums.npy"
PLDDT_TABLE_UNIT: float = 0.0001

# Position of each amino acid in AMINO_ACIDS, used to number feature windows.
RESIDUE_INDEX: dict[str, int] = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}

# The scaler metrics are a (42, 10) array with one row per feature. The columns
# are the fitted attributes of a scikit-learn StandardScaler (mean_, var_,
# scale_, n_samples_seen_) followed by those of a MinMaxScaler fit on the
# standardized features (min_, data_min_, data_max_, scale_, n_samples_seen_,
# data_range_). Only the four columns below are needed to scale new features.
SCALER_COLUMN_MEAN: int = 0
SCALER_COLUMN_STANDARD_DEVIATION: int = 2
SCALER_COLUMN_DATA_MIN: int = 5
SCALER_COLUMN_DATA_RANGE: int = 9
NUMBER_OF_SCALER_COLUMNS: int = 10


def get_data_path(file_name: str) -> str:
    """
    Get the path to one of the data files shipped with tadatorch.

    Parameters
    ----------
    file_name : str
        Name of a file in the tadatorch data directory.

    Returns
    -------
    str
        Path to the file.
    """
    return str(importlib.resources.files("tadatorch") / "data" / file_name)


def load_scaler_metrics() -> np.ndarray:
    """
    Load the metrics used to scale features before prediction.

    Returns
    -------
    np.ndarray
        Array of shape (42, 10) with one row per feature. See the comments on
        the SCALER_COLUMN constants in this module for what the columns are.
    """
    scaler_metrics: np.ndarray = np.load(get_data_path("scaler_metric.npy"))

    expected_shape = (NUMBER_OF_FEATURES, NUMBER_OF_SCALER_COLUMNS)
    assert scaler_metrics.shape == expected_shape, (
        f"Scaler metrics have shape {scaler_metrics.shape}, expected {expected_shape}."
    )

    return scaler_metrics


@functools.lru_cache(maxsize=1)
def load_plddt_table() -> np.ndarray:
    """
    Load the table of predicted pLDDT scores for every 5-residue window.

    One of the TADA features is the mean AlphaFold2 pLDDT score predicted for
    each feature window by alphaPredict. There are only 3.2 million possible
    5-residue windows, so rather than running the alphaPredict network (which
    was the slowest part of calculating features) the score of every one was
    calculated once and saved, using
    devtools/scripts/build_plddt_table.py. The table is read from disk the
    first time this function is called and is then kept in memory.

    Returns
    -------
    np.ndarray
        Array with one integer for every possible window, at the position
        given by ``number_window()``. Each value is the sum of the five
        per-residue scores, in units of 0.0001.
    """
    table: np.ndarray = np.load(get_data_path(PLDDT_TABLE_FILE))

    number_of_windows = len(AMINO_ACIDS) ** FEATURE_WINDOW_SIZE
    assert table.shape == (number_of_windows,), (
        f"The pLDDT table has shape {table.shape}, expected ({number_of_windows},)."
    )

    return table


def number_window(window: str) -> int:
    """
    Convert a feature window into the position of its entry in the pLDDT table.

    Parameters
    ----------
    window : str
        Amino acid sequence of one feature window (5 residues).

    Returns
    -------
    int
        The window read as a 5-digit number in base 20, where the digit for
        each residue is its position in AMINO_ACIDS. Every possible window
        gets a different number between 0 and 3,199,999.
    """
    number = 0
    for residue in window:
        number = number * len(AMINO_ACIDS) + RESIDUE_INDEX[residue]

    return number


def split_into_feature_windows(sequence: str) -> list[str]:
    """
    Split a sequence into the short overlapping windows features are
    calculated over.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    list of str
        The 5-residue windows of the sequence, where each window starts one
        residue after the previous one.
    """
    last_window_start = len(sequence) - FEATURE_WINDOW_SIZE

    return [
        sequence[start : start + FEATURE_WINDOW_SIZE]
        for start in range(0, last_window_start + 1, FEATURE_WINDOW_STEP)
    ]


@functools.lru_cache(maxsize=WINDOW_FEATURE_CACHE_SIZE)
def calculate_window_features(window: str) -> tuple[float, ...]:
    """
    Calculate the features that depend only on a single feature window.

    The features for recently seen windows are cached, because the same
    windows come up again and again when TADA is slid along a protein.

    Parameters
    ----------
    window : str
        Amino acid sequence of one feature window (5 residues).

    Returns
    -------
    tuple of float
        The 40 window-specific features. In order these are: mean
        Kyte-Doolittle hydropathy, mean Wimley-White hydropathy, net charge
        per residue, fraction of disorder promoting residues, fraction of
        charged residues, mean net charge, fraction of negative residues and
        fraction of positive residues (8 values, calculated with sparrow),
        the number of residues in each of the RESIDUE_GROUPS (11 values), the
        mean predicted AlphaFold2 pLDDT score (1 value, looked up from the
        alphaPredict table; this feature is what the original TADA code
        called 'sstructure') and the number of each of the 20 amino acids (20
        values).
    """
    net_charge_per_residue = physical_properties.net_charge_per_residue(window)
    physicochemical_properties = [
        physical_properties.mean_residue_scale(window, sparrow_amino_acids.AA_hydro_KD),
        physical_properties.mean_residue_scale(window, WIMLEY_WHITE_SCALE),
        net_charge_per_residue,
        physical_properties.fraction_disorder_promoting(window),
        physical_properties.fraction_charged(window),
        abs(net_charge_per_residue),
        physical_properties.fraction_negative(window),
        physical_properties.fraction_positive(window),
    ]

    group_counts = [
        sum(residue in group_residues for residue in window)
        for group_residues in RESIDUE_GROUPS.values()
    ]

    plddt_sum = int(load_plddt_table()[number_window(window)])
    mean_predicted_plddt = plddt_sum * PLDDT_TABLE_UNIT / len(window)

    amino_acid_counts = [window.count(amino_acid) for amino_acid in AMINO_ACIDS]

    return (
        *physicochemical_properties,
        *group_counts,
        mean_predicted_plddt,
        *amino_acid_counts,
    )


def calculate_patterning(
    sequences: list[str], group_1: list[str], group_2: list[str]
) -> np.ndarray:
    """
    Calculate how segregated two groups of residues are in each sequence.

    This is kappa for the two groups, calculated by sparrow for all of the
    sequences at once and averaged over the two blob sizes.

    Parameters
    ----------
    sequences : list of str
        Amino acid sequences, which must all be the same length.

    group_1 : list of str
        One-letter codes of the first group of residues.

    group_2 : list of str
        One-letter codes of the second group of residues. If this is empty,
        the second group is every residue that is not in the first.

    Returns
    -------
    np.ndarray
        The patterning of each sequence, between 0 (well mixed) and 1
        (segregated). The value is -1 where patterning is undefined: when two
        groups are given this is sequences that do not have residues from
        both, and when one group is given it is sequences with no residues
        from it.
    """
    # the final argument tells sparrow to cap values above 1 at exactly 1
    per_blob_size = [
        sparrow_patterning.kappa_x_batch(sequences, group_1, group_2, blob_size, 1)
        for blob_size in KAPPA_BLOB_SIZES
    ]

    patterning: np.ndarray = np.mean(per_blob_size, axis=0)
    return patterning


def calculate_kappa(sequences: list[str]) -> np.ndarray:
    """
    Calculate kappa, the patterning of oppositely charged residues.

    Parameters
    ----------
    sequences : list of str
        Amino acid sequences, which must all be the same length.

    Returns
    -------
    np.ndarray
        Kappa of each sequence (Das & Pappu 2013, Proc. Natl. Acad. Sci. 110,
        13392-13397), which is -1 for sequences that do not have both
        positive and negative residues.
    """
    return calculate_patterning(sequences, KAPPA_POSITIVE_RESIDUES, KAPPA_NEGATIVE_RESIDUES)


def calculate_omega(sequences: list[str]) -> np.ndarray:
    """
    Calculate Omega, the patterning of charged and proline residues with
    respect to all other residues.

    Parameters
    ----------
    sequences : list of str
        Amino acid sequences, which must all be the same length.

    Returns
    -------
    np.ndarray
        Omega of each sequence (Martin et al. 2016, J. Am. Chem. Soc. 138,
        15323-15335), which is -1 for sequences with no charged or proline
        residues.
    """
    return calculate_patterning(sequences, OMEGA_RESIDUES, [])


def collect_window_features(sequences: list[str]) -> np.ndarray:
    """
    Collect the window-specific features of every feature window of every
    sequence.

    The features of each distinct window are only calculated (or looked up)
    once, however many times the window comes up in the sequences.

    Parameters
    ----------
    sequences : list of str
        Amino acid sequences that are all 40 residues long.

    Returns
    -------
    np.ndarray
        Array of shape (number of sequences, 36, 40) holding the 40
        window-specific features of each of the 36 feature windows of each
        sequence.
    """
    windows_per_sequence = [split_into_feature_windows(sequence) for sequence in sequences]

    distinct_windows = sorted({window for windows in windows_per_sequence for window in windows})
    distinct_features = np.array([calculate_window_features(window) for window in distinct_windows])

    # for every feature window of every sequence, the row of its features
    row_of_window = {window: row for row, window in enumerate(distinct_windows)}
    rows = np.array(
        [[row_of_window[window] for window in windows] for windows in windows_per_sequence]
    )

    window_features: np.ndarray = distinct_features[rows]
    return window_features


def create_features(sequences: list[str]) -> np.ndarray:
    """
    Calculate the (unscaled) features for a list of 40 amino acid sequences.

    Parameters
    ----------
    sequences : list of str
        Amino acid sequences. Every sequence must be exactly 40 residues long
        and only contain the 20 standard amino acids (uppercase).

    Returns
    -------
    np.ndarray
        Array of shape (number of sequences, 36, 42). The second axis is the
        36 feature windows along each sequence and the third axis is the
        features. The first two features are kappa and Omega, which describe
        the whole sequence so are the same for all of its windows. The other
        40 are the window-specific features described in
        ``calculate_window_features()``.

    Raises
    ------
    ValueError
        If no sequences are passed, or if any sequence is not 40 residues long
        or contains invalid residues.
    """
    if len(sequences) == 0:
        raise ValueError("At least one sequence is needed to create features.")

    for sequence in sequences:
        validate_sequence(sequence)
        if len(sequence) != TADA_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequences must be exactly {TADA_SEQUENCE_LENGTH} amino acids long "
                f"to create features, but '{sequence}' is {len(sequence)} long."
            )

    features = np.empty((len(sequences), NUMBER_OF_FEATURE_WINDOWS, NUMBER_OF_FEATURES))

    # kappa and Omega describe the patterning of the whole sequence, so every
    # window gets the same value. Both are -1 for sequences where they are
    # undefined, and those values are used as they are.
    features[:, :, 0] = calculate_kappa(sequences)[:, np.newaxis]
    features[:, :, 1] = calculate_omega(sequences)[:, np.newaxis]
    features[:, :, 2:] = collect_window_features(sequences)

    return features


def scale_features(features: np.ndarray, scaler_metrics: np.ndarray) -> np.ndarray:
    """
    Scale features so they match the features TADA was trained on.

    Each feature is first standardized (the training mean is subtracted and
    the result is divided by the training standard deviation). The
    standardized values are then min-max scaled using the minimum and range
    the standardized features had in the training data.

    Parameters
    ----------
    features : np.ndarray
        Unscaled features with shape (number of sequences, 36, 42), as
        returned by ``create_features()``.

    scaler_metrics : np.ndarray
        Scaler metrics with shape (42, 10), as returned by
        ``load_scaler_metrics()``.

    Returns
    -------
    np.ndarray
        Scaled features with the same shape as ``features``.
    """
    assert features.shape[-1] == scaler_metrics.shape[0], (
        f"Got {features.shape[-1]} features but scaler metrics for "
        f"{scaler_metrics.shape[0]} features."
    )

    # one value per feature, which numpy broadcasts along the last axis
    mean = scaler_metrics[:, SCALER_COLUMN_MEAN]
    standard_deviation = scaler_metrics[:, SCALER_COLUMN_STANDARD_DEVIATION]
    data_min = scaler_metrics[:, SCALER_COLUMN_DATA_MIN]
    data_range = scaler_metrics[:, SCALER_COLUMN_DATA_RANGE]

    standardized_features = (features - mean) / standard_deviation
    scaled_features: np.ndarray = (standardized_features - data_min) / data_range

    return scaled_features
