"""User-facing functions for predicting TAD scores."""

import os

import protfasta

from tadatorch.backend.constants import TADA_SEQUENCE_LENGTH
from tadatorch.backend.domains import (
    DEFAULT_GAP_SIZE,
    DEFAULT_METHOD,
    DEFAULT_MIN_AD_SIZE,
    DEFAULT_SCORE_THRESHOLD,
    ActivationDomain,
    find_activation_domains,
)
from tadatorch.backend.predictor import predict_tada as _predict_tada
from tadatorch.backend.utils import (
    make_sequences_constant_length,
    map_sequences_to_prediction,
    sliding_window,
    validate_sequence,
    verbose_warning_message,
)

SAFE_MODE_ERROR_MESSAGE: str = (
    f"Not all sequences are at least {TADA_SEQUENCE_LENGTH} amino acids long. TADA "
    f"was not made for sequences under {TADA_SEQUENCE_LENGTH} amino acids. You can "
    "still make these predictions by setting safe_mode=False, but use this feature "
    "with extreme caution!"
)


def read_fasta_sequences(path_to_fasta: str) -> dict[str, str]:
    """
    Read the sequences in a FASTA file.

    Non-standard residues are converted to standard amino acids by protfasta
    (B to N, U to C, X to G and Z to Q, with '*', '-' and spaces removed), and
    lowercase sequences are converted to uppercase.

    Parameters
    ----------
    path_to_fasta : str
        Path to a FASTA file.

    Returns
    -------
    dict
        Dictionary where each key is the name of a sequence (its full FASTA
        header) and each value is the sequence.

    Raises
    ------
    ValueError
        If the path does not exist.
    """
    if not os.path.exists(path_to_fasta):
        raise ValueError(f"Path does not exist: {path_to_fasta}")

    names_to_sequences = protfasta.read_fasta(path_to_fasta, invalid_sequence_action="convert")

    # read_fasta only returns a list when return_list=True is passed
    assert isinstance(names_to_sequences, dict), "Expected protfasta to return a dictionary."

    return names_to_sequences


def predict(
    sequences: str | list[str],
    overlap_length: int = TADA_SEQUENCE_LENGTH - 1,
    pad: str = "GS",
    approach: str = "even",
    verbose: bool = True,
    safe_mode: bool = True,
    seed: int | None = None,
) -> dict[str, list[list[str | float]]]:
    """
    Predict TAD scores for a sequence or a list of sequences.

    TADA makes predictions on 40 amino acid sequences. Sequences longer than
    40 amino acids are broken into overlapping 40 amino acid windows and a
    score is returned for each window. Sequences shorter than 40 amino acids
    are padded to 40 amino acids, but only if safe_mode is set to False.

    Parameters
    ----------
    sequences : str or list of str
        A single sequence or a list of sequences to predict TAD scores for.
        Sequences can only contain the 20 standard amino acids (uppercase).

    overlap_length : int
        The number of residues shared by consecutive windows when a sequence
        is longer than 40 amino acids. Default is 39.

    pad : str
        Which residues sequences shorter than 40 amino acids are padded with.
        Options are 'GS' or 'random'. 'GS' pads with a random selection of G
        and S, while 'random' pads with a random selection of all 20 amino
        acids. Default is 'GS'.

    approach : str
        Where padding is added to sequences shorter than 40 amino acids.
        Options are 'even', 'N' or 'C'. 'even' pads both ends of the sequence,
        'N' pads only the N-terminus and 'C' pads only the C-terminus. Default
        is 'even'.

    verbose : bool
        Whether to print a warning when sequences are not all 40 amino acids
        long. Default is True.

    safe_mode : bool
        Whether to run in safe mode. Safe mode raises an exception if any
        sequence is under 40 amino acids. Default is True.

    seed : int or None
        Seed for the random number generator used to choose padding residues.
        Set this to get the same padding (and so the same scores) each time
        for sequences shorter than 40 amino acids. If None the padding is
        different every time. Default is None.

    Returns
    -------
    dict
        Dictionary where each key is an input sequence and each value is a
        list of [sequence, score] pairs. The sequence in each pair is the
        exact 40 amino acid sequence the prediction was made on (a window of,
        or a padded version of, the input sequence) and the score is the TAD
        score for that sequence.

    Raises
    ------
    ValueError
        If no sequences are passed, if a sequence contains invalid residues,
        or if safe_mode is True and a sequence is under 40 amino acids.
    """
    if isinstance(sequences, str):
        sequences = [sequences]

    if len(sequences) == 0:
        raise ValueError("At least one sequence is needed to make predictions.")

    for sequence in sequences:
        validate_sequence(sequence)

    sequence_lengths = [len(sequence) for sequence in sequences]

    if safe_mode and min(sequence_lengths) < TADA_SEQUENCE_LENGTH:
        raise ValueError(SAFE_MODE_ERROR_MESSAGE)

    if verbose and any(length != TADA_SEQUENCE_LENGTH for length in sequence_lengths):
        print(verbose_warning_message(overlap_length=overlap_length, pad=pad, approach=approach))

    # pad or window every sequence so we only predict on 40 amino acid sequences
    sequence_dict = make_sequences_constant_length(
        sequences, overlap_length=overlap_length, pad=pad, approach=approach, seed=seed
    )
    padded_or_windowed_sequences, sequence_to_indices = map_sequences_to_prediction(sequence_dict)

    scores = _predict_tada(padded_or_windowed_sequences)

    # map the scores back to the sequence they were generated from
    final_dict: dict[str, list[list[str | float]]] = {}
    for sequence, indices in sequence_to_indices.items():
        final_dict[sequence] = [
            [padded_or_windowed_sequences[index], scores[index]] for index in indices
        ]

    return final_dict


def predict_from_fasta(
    path_to_fasta: str,
    overlap_length: int = TADA_SEQUENCE_LENGTH - 1,
    pad: str = "GS",
    approach: str = "even",
    verbose: bool = True,
    safe_mode: bool = True,
    seed: int | None = None,
) -> dict[str, list[str | list[list[str | float]]]]:
    """
    Predict TAD scores for the sequences in a FASTA file.

    Non-standard residues in the FASTA file are converted to standard amino
    acids by protfasta (B to N, U to C, X to G and Z to Q, with '*', '-' and
    spaces removed) before predictions are made.

    Parameters
    ----------
    path_to_fasta : str
        Path to a FASTA file.

    overlap_length : int
        The number of residues shared by consecutive windows when a sequence
        is longer than 40 amino acids. Default is 39.

    pad : str
        Which residues sequences shorter than 40 amino acids are padded with.
        Options are 'GS' or 'random'. 'GS' pads with a random selection of G
        and S, while 'random' pads with a random selection of all 20 amino
        acids. Default is 'GS'.

    approach : str
        Where padding is added to sequences shorter than 40 amino acids.
        Options are 'even', 'N' or 'C'. 'even' pads both ends of the sequence,
        'N' pads only the N-terminus and 'C' pads only the C-terminus. Default
        is 'even'.

    verbose : bool
        Whether to print a warning when sequences are not all 40 amino acids
        long. Default is True.

    safe_mode : bool
        Whether to run in safe mode. Safe mode raises an exception if any
        sequence is under 40 amino acids. Default is True.

    seed : int or None
        Seed for the random number generator used to choose padding residues.
        If None the padding is different every time. Default is None.

    Returns
    -------
    dict
        Dictionary where each key is the name of a sequence in the FASTA file
        and each value is a two-element list. The first element is the
        sequence and the second element is the list of [sequence, score] pairs
        for that sequence (see ``predict()``).

    Raises
    ------
    ValueError
        If the path does not exist, or if safe_mode is True and a sequence is
        under 40 amino acids.
    """
    names_to_sequences = read_fasta_sequences(path_to_fasta)

    predictions = predict(
        list(names_to_sequences.values()),
        overlap_length=overlap_length,
        pad=pad,
        approach=approach,
        verbose=verbose,
        safe_mode=safe_mode,
        seed=seed,
    )

    # map sequence names to predictions
    final_dict: dict[str, list[str | list[list[str | float]]]] = {}
    for name, sequence in names_to_sequences.items():
        final_dict[name] = [sequence, predictions[sequence]]

    return final_dict


def predict_activation_domains(
    sequence: str,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    method: str = DEFAULT_METHOD,
    gap_size: int = DEFAULT_GAP_SIZE,
    min_ad_size: int = DEFAULT_MIN_AD_SIZE,
) -> list[ActivationDomain]:
    """
    Predict where the activation domains are in a sequence.

    A TAD score is predicted for every 40 amino acid window of the sequence
    (so each window starts one residue after the previous one). Note that
    TADA only scores windows, so where a domain starts and stops is a choice
    made by tadatorch, and there are two ways of making it.

    With the default 'whole_window' method, an activation domain is a
    contiguous region of the sequence in which every residue is covered by at
    least one window with a TAD score at or above the threshold. Domains are
    always at least 40 residues long.

    With the 'central_residues' method, only the two residues at the center
    of each window at or above the threshold are marked. Gaps between marked
    regions shorter than ``gap_size`` are then joined, and regions shorter
    than ``min_ad_size`` are removed. This gives tighter domains. No window
    is centered on the first or last 19 residues of a sequence, so a domain
    at an end of the sequence is extended to the terminus, but only if the
    ``min_ad_size`` residues nearest that end that a window can mark were all
    marked (so that one window at the very end of a sequence is not enough
    to call a domain there).

    Parameters
    ----------
    sequence : str
        Amino acid sequence at least 40 residues long. The sequence can only
        contain the 20 standard amino acids (uppercase).

    threshold : float
        Windows with a TAD score at or above this value (between 0 and 1) are
        treated as activation domains. Default is 0.5, which is the threshold
        used to classify TADs in the TADA paper.

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
        C-terminus. Each one holds the start and end of the domain (1-indexed
        and inclusive), its sequence, the number of windows at or above the
        threshold, and the mean and max TAD score of the windows belonging to
        it. The list is empty if there are none.

    Raises
    ------
    ValueError
        If the sequence contains invalid residues or is under 40 amino acids,
        or if the threshold, method, gap_size or min_ad_size are not valid.
    """
    validate_sequence(sequence)

    if len(sequence) < TADA_SEQUENCE_LENGTH:
        raise ValueError(
            f"Activation domains can only be predicted for sequences at least "
            f"{TADA_SEQUENCE_LENGTH} amino acids long, but got {len(sequence)}."
        )

    # every possible 40 amino acid window, so windows[i] is sequence[i:i + 40]
    windows = sliding_window(
        sequence, window_length=TADA_SEQUENCE_LENGTH, overlap=TADA_SEQUENCE_LENGTH - 1
    )
    window_scores = _predict_tada(windows)

    return find_activation_domains(
        sequence,
        window_scores,
        threshold=threshold,
        method=method,
        gap_size=gap_size,
        min_ad_size=min_ad_size,
    )
