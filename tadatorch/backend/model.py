"""
PyTorch implementation of the TADA network.

The architecture reproduces, layer for layer, the Keras model defined in
TADA_T2 (https://github.com/ryanemenecker/TADA_T2), which is itself a
TensorFlow2 port of the original TADA model written by Lisa Van den Broeck
(https://github.com/LisaVdB/TADA).
"""

import torch
from torch import nn

from tadatorch.backend.constants import NUMBER_OF_FEATURE_WINDOWS, NUMBER_OF_FEATURES

# Number of classes the network predicts: index 0 is 'is a TAD' and index 1 is
# 'is not a TAD'.
NUMBER_OF_CLASSES: int = 2

# The network has two convolutional layers, each of which shortens the window
# axis because no padding is applied.
NUMBER_OF_CONVOLUTIONS: int = 2


class Attention(nn.Module):
    """
    Attention layer that re-weights each position along the window axis.

    A single score is calculated for every position from its channels. The
    scores are converted to weights with a softmax over positions, and the
    input is then multiplied by those weights. The output therefore has the
    same shape as the input (this matches the ``return_sequences=True``
    behavior of the Keras layer used by TADA).

    Parameters
    ----------
    number_of_positions : int
        Length of the window axis of the input. The bias has one value per
        position, so this is fixed when the layer is built.

    number_of_channels : int
        Number of channels at each position of the input.
    """

    def __init__(self, number_of_positions: int, number_of_channels: int) -> None:
        super().__init__()

        # weight and bias shapes match the Keras layer (att_weight and
        # att_bias) so the trained values can be copied across unchanged
        self.weight = nn.Parameter(torch.empty(number_of_channels, 1))
        self.bias = nn.Parameter(torch.zeros(number_of_positions, 1))

        # Keras initialized the weight with glorot_uniform and the bias with zeros
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply attention weights to the input.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, number_of_positions, number_of_channels).

        Returns
        -------
        torch.Tensor
            The re-weighted input, with the same shape as ``inputs``.
        """
        # one score per position: (batch, number_of_positions, 1)
        scores = torch.tanh(torch.matmul(inputs, self.weight) + self.bias)

        # softmax along the window axis so the weights at all positions sum to 1
        attention_weights = torch.softmax(scores, dim=1)

        return inputs * attention_weights


def apply_convolution(convolution: nn.Conv1d, inputs: torch.Tensor) -> torch.Tensor:
    """
    Apply an unpadded 1D convolution (with a step of 1) as matrix multiplications.

    This gives the same result as calling the convolution itself, but is
    around 50 times faster on a CPU for the small inputs TADA uses, where the
    convolutions were otherwise three quarters of the time taken by the
    network.
    It also works directly on (batch, positions, channels) tensors, which is
    the layout of the features and of every other layer in the network.

    Parameters
    ----------
    convolution : nn.Conv1d
        The convolution, which holds the weights and bias.

    inputs : torch.Tensor
        Tensor of shape (batch, positions, input channels).

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, positions - kernel size + 1, output channels).
    """
    assert convolution.bias is not None, "The convolution must have a bias."

    kernel_size = convolution.kernel_size[0]
    number_of_outputs = inputs.shape[1] - kernel_size + 1

    # each output position is the sum over the kernel of the input at one
    # offset multiplied by the weights for that offset, plus the bias. The
    # weights have shape (output channels, input channels, kernel size).
    outputs: torch.Tensor = convolution.bias
    for offset in range(kernel_size):
        shifted_inputs = inputs[:, offset : offset + number_of_outputs, :]
        outputs = outputs + torch.matmul(shifted_inputs, convolution.weight[:, :, offset].T)

    return outputs


class TadaModel(nn.Module):
    """
    The TADA network.

    The network is two 1D convolutions, an attention layer, two bidirectional
    LSTMs and a final dense layer with a softmax. The default arguments
    reproduce the network that the shipped weights were trained with, so they
    should only be changed if you are training a new network.

    Note that the original Keras model applied an L1/L2 penalty to the kernel
    of the first convolution. That penalty is a term in the training loss and
    has no effect on predictions, so it is not part of this module.

    Parameters
    ----------
    number_of_windows : int
        Number of feature windows per sequence (length of the window axis of
        the input). Default is 36.

    number_of_features : int
        Number of features per window. Default is 42.

    kernel_size : int
        Kernel size of both convolutions. Default is 2.

    filters : int
        Number of filters (output channels) in both convolutions. Default is
        100.

    dropout : float
        Dropout probability applied after each convolution. Dropout is only
        active in training mode. Default is 0.3.

    bilstm_output_size : int
        Hidden size of each direction of the two bidirectional LSTMs. Default
        is 100.
    """

    def __init__(
        self,
        number_of_windows: int = NUMBER_OF_FEATURE_WINDOWS,
        number_of_features: int = NUMBER_OF_FEATURES,
        kernel_size: int = 2,
        filters: int = 100,
        dropout: float = 0.3,
        bilstm_output_size: int = 100,
    ) -> None:
        super().__init__()

        # each unpadded convolution shortens the window axis by (kernel_size - 1)
        number_of_positions = number_of_windows - NUMBER_OF_CONVOLUTIONS * (kernel_size - 1)
        if number_of_positions < 1:
            raise ValueError(
                f"A kernel_size of {kernel_size} is too large for an input with "
                f"{number_of_windows} windows."
            )

        self.conv_1 = nn.Conv1d(number_of_features, filters, kernel_size)
        self.conv_2 = nn.Conv1d(filters, filters, kernel_size)

        # Keras' 'gelu' activation is the exact (erf-based) GELU, which is also
        # the PyTorch default
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.attention = Attention(number_of_positions, filters)

        # both directions are concatenated, so the output of each bidirectional
        # LSTM has 2 * bilstm_output_size channels
        self.bilstm_1 = nn.LSTM(filters, bilstm_output_size, batch_first=True, bidirectional=True)
        self.bilstm_2 = nn.LSTM(
            2 * bilstm_output_size,
            bilstm_output_size,
            batch_first=True,
            bidirectional=True,
        )

        self.dense = nn.Linear(2 * bilstm_output_size, NUMBER_OF_CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calculate class probabilities from scaled features.

        Parameters
        ----------
        features : torch.Tensor
            Scaled features with shape (batch, number_of_windows,
            number_of_features).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 2) where each row sums to 1. Column 0 is
            the probability the sequence is a TAD (the TAD score) and column 1
            is the probability that it is not.
        """
        hidden = self.dropout(self.activation(apply_convolution(self.conv_1, features)))
        hidden = self.dropout(self.activation(apply_convolution(self.conv_2, hidden)))

        hidden = self.attention(hidden)

        hidden, _ = self.bilstm_1(hidden)

        # the second LSTM only passes on its final state. For the forward
        # direction this is the state after the last position, and for the
        # backward direction it is the state after the first position, which is
        # what Keras returns for Bidirectional(LSTM(return_sequences=False)).
        _, (final_hidden_states, _) = self.bilstm_2(hidden)
        forward_final_state = final_hidden_states[0]
        backward_final_state = final_hidden_states[1]
        hidden = torch.cat([forward_final_state, backward_final_state], dim=1)

        return torch.softmax(self.dense(hidden), dim=1)
