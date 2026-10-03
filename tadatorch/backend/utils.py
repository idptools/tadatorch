"""
Utilities for validating sequences and for converting sequences of any length
into the 40 amino acid sequences that TADA makes predictions on.
"""

import random

from tadatorch.backend.constants import AMINO_ACIDS, TADA_SEQUENCE_LENGTH

# options for which residues are used to pad short sequences
VALID_PAD_OPTIONS: tuple[str, ...] = ("random", "GS")

# options for which end(s) of a short sequence the padding is added to
VALID_APPROACH_OPTIONS: tuple[str, ...] = ("even", "N", "C")

# residues used for padding when pad='GS'
GS_PAD_RESIDUES: tuple[str, ...] = ("G", "S")


def validate_sequence(sequence: str) -> None:
    """
    Check that a sequence is a string made up of the 20 standard amino acids.

    Sequences must be uppercase. Several of the TADA features are calculated
    by matching residues against uppercase one-letter codes, so a lowercase
    sequence would otherwise silently generate incorrect features.

    Parameters
    ----------
    sequence : str
        Amino acid sequence to check.

    Raises
    ------
    ValueError
        If the sequence is not a string, or if it contains anything other
        than the 20 standard (uppercase) amino acids.
    """
    if not isinstance(sequence, str):
        raise ValueError(f"Sequences must be strings, but got {type(sequence).__name__}.")

    invalid_residues = sorted(set(sequence) - set(AMINO_ACIDS))
    if len(invalid_residues) > 0:
        raise ValueError(
            f"Invalid residue(s) {invalid_residues} found in sequence '{sequence}'. "
            "Sequences can only contain the 20 standard amino acids (uppercase)."
        )


def sliding_window(sequence: str, window_length: int, overlap: int) -> list[str]:
    """
    Break a sequence up into windows of a fixed length.

    Windows start at the N-terminus and each window overlaps the previous one
    by ``overlap`` residues. Note that if the sequence does not divide evenly
    into windows with the requested overlap, the residues left over at the
    C-terminus are not included in any window.

    Parameters
    ----------
    sequence : str
        The sequence to break up.

    window_length : int
        The length of each window (number of residues).

    overlap : int
        The number of residues shared by consecutive windows.

    Returns
    -------
    list of str
        The windows, in order from the N- to the C-terminus.

    Raises
    ------
    ValueError
        If the window length is not between 1 and the length of the sequence,
        or the overlap is not between 0 and window_length - 1.
    """
    if window_length > len(sequence) or window_length <= 0:
        raise ValueError(
            "Window length must be a positive integer and less than or equal to "
            "the length of the input sequence."
        )

    if overlap < 0 or overlap >= window_length:
        raise ValueError("Overlap must be a non-negative integer less than the window length.")

    step = window_length - overlap
    last_window_start = len(sequence) - window_length

    return [
        sequence[start : start + window_length] for start in range(0, last_window_start + 1, step)
    ]


def pad_sequence(
    input_sequence: str,
    pad: str = "GS",
    objective_length: int = TADA_SEQUENCE_LENGTH,
    approach: str = "even",
    random_generator: random.Random | None = None,
) -> str:
    """
    Pad a sequence with randomly chosen residues until it reaches a set length.

    Sequences that are already at (or over) the objective length are returned
    unchanged.

    Parameters
    ----------
    input_sequence : str
        The sequence to pad.

    pad : str
        Which residues to pad with. Options are 'GS' or 'random'. 'GS' pads
        with a random selection of G and S, while 'random' pads with a random
        selection of all 20 amino acids. Default is 'GS'.

    objective_length : int
        The length (number of residues) to pad the sequence to. Default is 40.

    approach : str
        Where the padding is added. Options are 'even', 'N' or 'C'. 'even'
        splits the padding between both ends of the sequence (if the padding
        is an odd length the extra residue goes on the C-terminus), 'N' pads
        only the N-terminus and 'C' pads only the C-terminus. Default is
        'even'.

    random_generator : random.Random or None
        Random number generator used to choose the padding residues. Pass a
        seeded generator to make the padding reproducible. If None, a new
        (unseeded) generator is used. Default is None.

    Returns
    -------
    str
        The padded sequence.

    Raises
    ------
    ValueError
        If pad or approach are not valid options, or if objective_length is
        negative.
    """
    if pad not in VALID_PAD_OPTIONS:
        raise ValueError("Pad must be either random or GS.")
    if approach not in VALID_APPROACH_OPTIONS:
        raise ValueError("Approach must be either even, N, or C.")
    if objective_length < 0:
        raise ValueError("Objective length must be a positive integer.")

    if len(input_sequence) >= objective_length:
        return input_sequence

    if random_generator is None:
        random_generator = random.Random()

    pad_residues = GS_PAD_RESIDUES if pad == "GS" else AMINO_ACIDS

    number_to_pad = objective_length - len(input_sequence)
    padding = "".join(random_generator.choice(pad_residues) for _ in range(number_to_pad))

    if approach == "N":
        return padding + input_sequence
    if approach == "C":
        return input_sequence + padding

    # approach is 'even', so split the padding across the two termini
    number_n_terminal = number_to_pad // 2
    return padding[:number_n_terminal] + input_sequence + padding[number_n_terminal:]


def make_sequences_constant_length(
    sequence_list: list[str],
    objective_length: int = TADA_SEQUENCE_LENGTH,
    overlap_length: int = TADA_SEQUENCE_LENGTH - 1,
    pad: str = "GS",
    approach: str = "even",
    seed: int | None = None,
) -> dict[str, list[str]]:
    """
    Convert every sequence into one or more sequences of a constant length.

    Sequences shorter than the objective length are padded using
    ``pad_sequence()``, sequences longer than the objective length are broken
    into overlapping windows using ``sliding_window()``, and sequences that
    are already the objective length are left as they are.

    Parameters
    ----------
    sequence_list : list of str
        The sequences to pad or window.

    objective_length : int
        The length (number of residues) every returned sequence will be.
        Default is 40.

    overlap_length : int
        The number of residues shared by consecutive windows when a sequence
        is longer than the objective length. Default is 39.

    pad : str
        Which residues short sequences are padded with. Options are 'GS' or
        'random' (see ``pad_sequence()``). Default is 'GS'.

    approach : str
        Where padding is added to short sequences. Options are 'even', 'N' or
        'C' (see ``pad_sequence()``). Default is 'even'.

    seed : int or None
        Seed for the random number generator used to choose padding residues.
        If None the padding is different every time. Default is None.

    Returns
    -------
    dict
        Dictionary where each key is an input sequence and each value is the
        list of constant-length sequences generated from it. Keys are in the
        order the sequences were passed in, and an input sequence that appears
        more than once only gets a single entry.
    """
    random_generator = random.Random(seed)

    constant_length_sequences: dict[str, list[str]] = {}
    for sequence in sequence_list:
        if len(sequence) > objective_length:
            constant_length_sequences[sequence] = sliding_window(
                sequence, objective_length, overlap_length
            )
        elif len(sequence) < objective_length:
            padded_sequence = pad_sequence(
                sequence,
                pad=pad,
                objective_length=objective_length,
                approach=approach,
                random_generator=random_generator,
            )
            constant_length_sequences[sequence] = [padded_sequence]
        else:
            constant_length_sequences[sequence] = [sequence]

    return constant_length_sequences


def map_sequences_to_prediction(
    sequence_dict: dict[str, list[str]],
) -> tuple[list[str], dict[str, list[int]]]:
    """
    Flatten the constant-length sequences into a single list for prediction.

    The network makes predictions on one flat list of sequences, so this
    function also keeps track of which positions in that list came from which
    of the original input sequences.

    Parameters
    ----------
    sequence_dict : dict
        Dictionary where each key is an input sequence and each value is the
        list of constant-length sequences generated from it (the output of
        ``make_sequences_constant_length()``).

    Returns
    -------
    list of str
        All of the constant-length sequences as one flat list.

    dict
        Dictionary where each key is an input sequence and each value is the
        list of indices into the flat list that belong to that sequence.
    """
    all_sequences: list[str] = []
    sequence_to_indices: dict[str, list[int]] = {}

    for sequence, constant_length_sequences in sequence_dict.items():
        first_index = len(all_sequences)
        all_sequences.extend(constant_length_sequences)
        sequence_to_indices[sequence] = list(range(first_index, len(all_sequences)))

    return all_sequences, sequence_to_indices


def verbose_warning_message(overlap_length: int, pad: str, approach: str) -> str:
    """
    Build the warning shown when sequences are not all 40 amino acids long.

    Parameters
    ----------
    overlap_length : int
        The number of residues shared by consecutive windows for sequences
        longer than 40 amino acids.

    pad : str
        Which residues short sequences are padded with. Options are 'GS' or
        'random' (see ``pad_sequence()``).

    approach : str
        Where padding is added to short sequences. Options are 'even', 'N' or
        'C' (see ``pad_sequence()``).

    Returns
    -------
    str
        The warning message.
    """
    if pad == "GS":
        pad_message = "random selection of G and S"
    else:
        pad_message = "random selection of all amino acids"

    if approach == "even":
        approach_message = "evenly on the N and C terminus"
    elif approach == "N":
        approach_message = "only at the N terminus"
    else:
        approach_message = "only at the C terminus"

    return (
        f"Warning: Not all sequences are {TADA_SEQUENCE_LENGTH} amino acids long.\n"
        f"Sequences shorter than {TADA_SEQUENCE_LENGTH} amino acids will be padded "
        f"with a {pad_message} {approach_message}.\n"
        f"Sequences longer than {TADA_SEQUENCE_LENGTH} amino acids will be windowed "
        f"to make sequences {TADA_SEQUENCE_LENGTH} amino acids in length with "
        f"{overlap_length} overlapping amino acids."
    )
