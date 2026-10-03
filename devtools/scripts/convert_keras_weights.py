"""
Convert the original TADA Keras weights (.hdf5) into a PyTorch state dict (.pt).

This script documents exactly how the weights shipped with tadatorch were
generated, and only needs to be re-run if the weights ever need regenerating.
It requires h5py, which is not a dependency of tadatorch itself.

Usage
-----
python convert_keras_weights.py <path to tada.14-0.02.hdf5> <path to output .pt>

The Keras weights file can be found in the data directory of TADA_T2
(https://github.com/ryanemenecker/TADA_T2).
"""

import argparse

import h5py
import numpy as np
import torch

from tadatorch.backend.model import TadaModel

# Where each pair of convolution weights lives in the Keras file, keyed by the
# name of the matching layer in TadaModel.
CONVOLUTION_LAYERS: dict[str, str] = {
    "conv_1": "conv1d/conv1d",
    "conv_2": "conv1d_1/conv1d_1",
}

# Where each LSTM cell lives in the Keras file, keyed by the name of the
# matching layer in TadaModel. Each bidirectional LSTM has a forward cell and
# a backward cell (given in that order).
BIDIRECTIONAL_LSTM_LAYERS: dict[str, tuple[str, str]] = {
    "bilstm_1": (
        "bidirectional/bidirectional/forward_lstm/lstm_cell_1",
        "bidirectional/bidirectional/backward_lstm/lstm_cell_2",
    ),
    "bilstm_2": (
        "bidirectional_1/bidirectional_1/forward_lstm_1/lstm_cell_4",
        "bidirectional_1/bidirectional_1/backward_lstm_1/lstm_cell_5",
    ),
}

# PyTorch names the weights of the forward and backward directions of a
# single-layer bidirectional LSTM with these suffixes.
LSTM_DIRECTION_SUFFIXES: tuple[str, str] = ("_l0", "_l0_reverse")


def read_keras_weight(keras_file: h5py.File, dataset_path: str) -> torch.Tensor:
    """
    Read a single weight array from the Keras weights file.

    Parameters
    ----------
    keras_file : h5py.File
        The open Keras weights (.hdf5) file.

    dataset_path : str
        Path to the weight array inside the file.

    Returns
    -------
    torch.Tensor
        The weight array as a tensor.
    """
    return torch.from_numpy(np.array(keras_file[dataset_path]))


def convert_convolution_weights(keras_file: h5py.File) -> dict[str, torch.Tensor]:
    """
    Convert the weights of the two 1D convolutions.

    Keras stores a Conv1D kernel as (kernel_size, input_channels, filters)
    whereas PyTorch expects (filters, input_channels, kernel_size), so the
    first and last axes are swapped.

    Parameters
    ----------
    keras_file : h5py.File
        The open Keras weights (.hdf5) file.

    Returns
    -------
    dict
        The PyTorch state dict entries for both convolutions.
    """
    state_dict = {}
    for torch_name, keras_path in CONVOLUTION_LAYERS.items():
        kernel = read_keras_weight(keras_file, f"{keras_path}/kernel:0")
        state_dict[f"{torch_name}.weight"] = kernel.permute(2, 1, 0)
        state_dict[f"{torch_name}.bias"] = read_keras_weight(keras_file, f"{keras_path}/bias:0")

    return state_dict


def convert_lstm_weights(keras_file: h5py.File) -> dict[str, torch.Tensor]:
    """
    Convert the weights of the two bidirectional LSTMs.

    Keras and PyTorch order the four LSTM gates in the same way (input,
    forget, cell, output), so the gates do not need re-ordering. However,
    Keras stores the kernels as (input_size, 4 * units) whereas PyTorch
    expects (4 * units, input_size), so the kernels are transposed. Keras
    also has a single bias per cell where PyTorch has two that are summed
    (bias_ih and bias_hh), so the Keras bias is used for bias_ih and bias_hh
    is set to zero.

    Parameters
    ----------
    keras_file : h5py.File
        The open Keras weights (.hdf5) file.

    Returns
    -------
    dict
        The PyTorch state dict entries for both bidirectional LSTMs.
    """
    state_dict = {}
    for torch_name, keras_cell_paths in BIDIRECTIONAL_LSTM_LAYERS.items():
        for suffix, keras_path in zip(LSTM_DIRECTION_SUFFIXES, keras_cell_paths):
            kernel = read_keras_weight(keras_file, f"{keras_path}/kernel:0")
            recurrent_kernel = read_keras_weight(keras_file, f"{keras_path}/recurrent_kernel:0")
            bias = read_keras_weight(keras_file, f"{keras_path}/bias:0")

            state_dict[f"{torch_name}.weight_ih{suffix}"] = kernel.T
            state_dict[f"{torch_name}.weight_hh{suffix}"] = recurrent_kernel.T
            state_dict[f"{torch_name}.bias_ih{suffix}"] = bias
            state_dict[f"{torch_name}.bias_hh{suffix}"] = torch.zeros_like(bias)

    return state_dict


def convert_keras_weights(keras_file: h5py.File) -> dict[str, torch.Tensor]:
    """
    Convert all of the TADA Keras weights into a PyTorch state dict.

    Parameters
    ----------
    keras_file : h5py.File
        The open Keras weights (.hdf5) file.

    Returns
    -------
    dict
        State dict that can be loaded by ``tadatorch.backend.model.TadaModel``.
    """
    state_dict = convert_convolution_weights(keras_file)
    state_dict.update(convert_lstm_weights(keras_file))

    # the attention weights have the same shape in Keras and PyTorch
    state_dict["attention.weight"] = read_keras_weight(
        keras_file, "attention/attention/att_weight:0"
    )
    state_dict["attention.bias"] = read_keras_weight(keras_file, "attention/attention/att_bias:0")

    # Keras stores a Dense kernel as (input_size, output_size) whereas PyTorch
    # expects (output_size, input_size)
    state_dict["dense.weight"] = read_keras_weight(keras_file, "dense/dense/kernel:0").T
    state_dict["dense.bias"] = read_keras_weight(keras_file, "dense/dense/bias:0")

    # permuted/transposed tensors are views, so make every tensor contiguous
    # before it is saved
    return {name: weight.contiguous() for name, weight in state_dict.items()}


def main() -> None:
    """Read the Keras weights, convert them and save the PyTorch state dict."""
    parser = argparse.ArgumentParser(
        description="Convert the TADA Keras weights (.hdf5) into a PyTorch state dict (.pt)."
    )
    parser.add_argument("keras_weights", help="Path to the TADA Keras weights (.hdf5) file.")
    parser.add_argument("output", help="Path the PyTorch state dict (.pt) is written to.")
    arguments = parser.parse_args()

    with h5py.File(arguments.keras_weights, "r") as keras_file:
        state_dict = convert_keras_weights(keras_file)

    # loading into the network (strict by default) confirms every weight in the
    # network was converted and has the right shape, and that nothing is left
    # over
    TadaModel().load_state_dict(state_dict)

    torch.save(state_dict, arguments.output)
    print(f"Converted {len(state_dict)} weight tensors and saved them to {arguments.output}")


if __name__ == "__main__":
    main()
