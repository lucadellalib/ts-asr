"""Structured state space sequence (S4) model.

Authors
* Luca Della Libera 2023
"""

# TODO: recurrent view for efficient inference
# TODO: causality
# TODO: double check GSS

from typing import Optional

import torch
from speechbrain.lobes.models.transformer.Conformer import ConvolutionModule as CM
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import PositionalwiseFeedForward
from speechbrain.nnet.normalization import LayerNorm
from torch import nn

try:
    from dss import DSS
except ImportError:
    from .dss import DSS


__all__ = ["S4Encoder"]


class ConvolutionModule(CM):
    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super(CM, self).__init__()

        self.causal = causal

        if self.causal:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:
            self.padding = "same"

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # Pointwise
            nn.Conv1d(input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias),
            nn.GLU(dim=1),
        )
        # Depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            # Pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )


class S4EncoderLayer(nn.Module):
    """S4 encoder layer.

    Arguments
    ---------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of S4 feed forward layer.
    kernel_size : int, optional
        Kernel size of convolution model.
    activation: torch.nn.Module
        Activation function used in each S4 layer.
    bias : bool, optional
        Whether bias should be used in the convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    s4_layer_type: type, optional
        The S4 layer type.
    s4_layer_kwargs: dict, optional
        The S4 layer keyword arguments.

    Example
    -------
    >>> import torch
    >>>
    >>>
    >>> input = torch.rand((8, 60, 512))
    >>> model = S4EncoderLayer(d_ffn=512, d_model=512, kernel_size=3)
    >>> output = model(input)
    >>> output[0].shape

    """

    def __init__(
        self,
        d_model,
        d_ffn,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        s4_layer_type=DSS,
        s4_layer_kwargs={},
    ):
        super().__init__()

        self.s4_layer = s4_layer_type(input_size=d_model, **s4_layer_kwargs,)

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x, src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src sequence per batch.

        """
        conv_mask = None
        if src_key_padding_mask is not None:
            x[src_key_padding_mask] = 0.0
            conv_mask = src_key_padding_mask[..., None]
        # Feed-forward module
        x = x + 0.5 * self.ffn_module1(x)
        # S4 module
        skip = x
        x = self.norm1(x)
        x = self.s4_layer(x)
        x = x + skip
        # Convolution module
        x = x + self.convolution_module(x, conv_mask)
        # Feed-forward module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x


class S4Encoder(nn.Module):
    """S4 encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Embedding dimension size.
    d_ffn : int
        Hidden size of S4 feed forward layer.
    kernel_size : int, optional
        Kernel size of convolution model.
    activation: torch.nn.Module
        Activation function used in each S4 layer.
    bias : bool, optional
        Whether bias should be used in the convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    s4_layer_type: type, optional
        The S4 layer type.
    s4_layer_kwargs: dict, optional
        The S4 layer keyword arguments.

    Example
    -------
    >>> import torch
    >>>
    >>>
    >>> input = torch.rand((8, 60, 512))
    >>> model = S4Encoder(1, 512, 512, 8)
    >>> output = model(input)
    >>> output.shape

    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        s4_layer_type=DSS,
        s4_layer_kwargs={},
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                S4EncoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    s4_layer_type=s4_layer_type,
                    s4_layer_kwargs=s4_layer_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self, src, src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src sequence per batch.

        """
        output = src
        for enc_layer in self.layers:
            output = enc_layer(output, src_key_padding_mask=src_key_padding_mask,)
        output = self.norm(output)

        return output


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 2048
    input_size = 512
    model = S4Encoder(1, 512, 512, 8)
    input = torch.randn(batch_size, seq_length, input_size)
    output = model(input)
    print(output.shape)
