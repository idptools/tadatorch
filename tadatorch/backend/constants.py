"""
Constants shared across the tadatorch backend.

These values are fixed by how the original TADA network was trained (see
Morffy, Van den Broeck et al. 2024, Nature 632, 166-173 and
https://github.com/LisaVdB/TADA), so they are not user-tunable; changing any
of them means the shipped weights no longer apply.
"""

# TADA was trained on 40 amino acid fragments, so every sequence passed to the
# network must be exactly this long.
TADA_SEQUENCE_LENGTH: int = 40

# Each 40 amino acid sequence is described by sliding a 5-residue window along
# it one residue at a time, and computing a set of features for every window.
FEATURE_WINDOW_SIZE: int = 5
FEATURE_WINDOW_STEP: int = 1

# Number of feature windows that fit in a single TADA sequence (36).
NUMBER_OF_FEATURE_WINDOWS: int = (
    TADA_SEQUENCE_LENGTH - FEATURE_WINDOW_SIZE
) // FEATURE_WINDOW_STEP + 1

# Number of features calculated for every feature window.
NUMBER_OF_FEATURES: int = 42

# The 20 standard amino acids. The order here matches the order in which the
# per-residue count features were laid out when TADA was trained, so it must
# not be changed.
AMINO_ACIDS: tuple[str, ...] = (
    "R", "K", "D", "E", "Q", "N", "H", "S", "T", "Y",
    "C", "W", "M", "A", "I", "L", "F", "V", "P", "G",
)  # fmt: skip
