"""
Code for running the TADA network on 40 amino acid sequences.
"""

import functools
import importlib.resources

import numpy as np
import torch

from tadatorch.backend.features import (
    create_features,
    load_scaler_metrics,
    scale_features,
)
from tadatorch.backend.model import TadaModel

# Number of sequences passed through the network at once. This has no effect
# on the predictions. The network is around 30% faster per sequence with
# batches of 256 than with batches of 32 (the default in Keras, which TADA_T2
# used), and there is no further gain from larger batches.
DEFAULT_BATCH_SIZE: int = 256

# The network is small and feature calculation dominates the run time, so
# predictions are always run on the CPU.
DEVICE: str = "cpu"


def get_model_path() -> str:
    """
    Get the path to the trained network weights.

    Returns
    -------
    str
        Path to the PyTorch weights (.pt) file shipped with tadatorch. These
        are the original TADA weights (tada.14-0.02.hdf5) converted from
        Keras with devtools/scripts/convert_keras_weights.py.
    """
    model_weights_path = importlib.resources.files("tadatorch") / "data" / "tada.14-0.02.pt"
    return str(model_weights_path)


@functools.lru_cache(maxsize=1)
def load_model() -> TadaModel:
    """
    Build the TADA network and load the trained weights.

    The network is cached, so the weights are only read from disk the first
    time this function is called.

    Returns
    -------
    TadaModel
        The trained network, in evaluation mode.
    """
    model = TadaModel()

    # weights_only=True means only tensors are unpickled from the weights file
    state_dict = torch.load(get_model_path(), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    # evaluation mode switches off dropout
    model.eval()

    return model


def predict_tada(
    sequences: list[str],
    return_both_values: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float] | np.ndarray:
    """
    Predict TAD scores for a list of 40 amino acid sequences.

    Parameters
    ----------
    sequences : list of str
        Sequences to predict TAD scores for. Every sequence must be exactly 40
        residues long and only contain the 20 standard amino acids.

    return_both_values : bool
        The network returns two values per sequence, one for each category
        (is a TAD or is not a TAD). If True both values are returned. If False
        only the first value is returned, which is the 'TAD' score used in the
        TADA paper. Default is False.

    batch_size : int
        Number of sequences passed through the network at once. Default is 256.

    Returns
    -------
    list of float or np.ndarray
        If return_both_values is False, a list with the TAD score for each
        input sequence. If return_both_values is True, an array of shape
        (number of sequences, 2) where the first column is the TAD score and
        the second column is 1 - TAD score.

    Raises
    ------
    ValueError
        If sequences is not a list or batch_size is not a positive integer.
    """
    if not isinstance(sequences, list):
        raise ValueError("Sequences must be input as a list!")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    # get scaled features. The network weights are float32, so the features
    # are converted to match.
    features = create_features(sequences)
    scaled_features = scale_features(features, load_scaler_metrics())
    feature_tensor = torch.from_numpy(scaled_features).to(torch.float32)

    model = load_model()

    # run predictions
    batch_predictions = []
    with torch.inference_mode():
        for feature_batch in torch.split(feature_tensor, batch_size):
            batch_predictions.append(model(feature_batch))
    predictions: np.ndarray = torch.cat(batch_predictions).numpy()

    assert predictions.shape[0] == len(sequences), (
        f"Got {predictions.shape[0]} predictions for {len(sequences)} sequences."
    )

    if return_both_values:
        return predictions
    return [float(tad_score) for tad_score in predictions[:, 0]]
