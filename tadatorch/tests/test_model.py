"""
Tests for the PyTorch implementation of the TADA network.
"""

import pytest
import torch

from tadatorch.backend.constants import NUMBER_OF_FEATURE_WINDOWS, NUMBER_OF_FEATURES
from tadatorch.backend.model import Attention, TadaModel
from tadatorch.backend.predictor import get_model_path, load_model

# seed used to make the random test inputs reproducible
RANDOM_SEED: int = 0


def make_random_features(number_of_sequences: int) -> torch.Tensor:
    """
    Make a reproducible tensor of random values with the shape of scaled features.

    Parameters
    ----------
    number_of_sequences : int
        Size of the batch axis.

    Returns
    -------
    torch.Tensor
        Tensor of shape (number_of_sequences, 36, 42) with values between 0 and 1.
    """
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    shape = (number_of_sequences, NUMBER_OF_FEATURE_WINDOWS, NUMBER_OF_FEATURES)
    return torch.rand(shape, generator=generator)


def test_attention_keeps_shape_and_weights_sum_to_one() -> None:
    """
    The attention layer should return a tensor the same shape as its input.
    With an input of all ones the output is the attention weights themselves,
    which should sum to 1 along the window axis.
    """
    number_of_positions = 34
    number_of_channels = 100
    attention = Attention(number_of_positions, number_of_channels)

    inputs = torch.ones(3, number_of_positions, number_of_channels)
    outputs = attention(inputs)

    assert outputs.shape == inputs.shape
    torch.testing.assert_close(outputs.sum(dim=1), torch.ones(3, number_of_channels))


def test_model_output_is_two_probabilities() -> None:
    """The network should return two probabilities per sequence that sum to 1."""
    model = TadaModel().eval()
    features = make_random_features(number_of_sequences=4)

    with torch.inference_mode():
        predictions = model(features)

    assert predictions.shape == (4, 2)
    assert bool((predictions >= 0).all())
    torch.testing.assert_close(predictions.sum(dim=1), torch.ones(4))


def test_shipped_weights_match_architecture() -> None:
    """
    Every weight in the shipped weights file should match a weight in the
    network (load_state_dict is strict, so raises if anything is missing,
    unexpected or the wrong shape).
    """
    state_dict = torch.load(get_model_path(), map_location="cpu", weights_only=True)
    TadaModel().load_state_dict(state_dict)


def test_loaded_model_is_cached_and_in_evaluation_mode() -> None:
    """The loaded network should be reused between calls and have dropout off."""
    model = load_model()

    assert model is load_model()
    assert not model.training


def test_predictions_do_not_depend_on_batch_composition() -> None:
    """A sequence should get the same prediction alone as it does in a batch."""
    model = load_model()
    features = make_random_features(number_of_sequences=3)

    with torch.inference_mode():
        batch_predictions = model(features)
        single_prediction = model(features[:1])

    torch.testing.assert_close(batch_predictions[:1], single_prediction, atol=1e-6, rtol=0)


def test_model_rejects_kernel_that_is_too_large() -> None:
    """A kernel larger than the input can support should fail when the network is built."""
    with pytest.raises(ValueError):
        TadaModel(number_of_windows=4, kernel_size=5)
