"""S4 model.

Authors
* Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/speechbrain/lobes/models/transformer/TransformerASR.py

# TODO: recurrent view for efficient inference
# TODO: causality
# TODO: double check GSS

from typing import Optional

import torch
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.Conformer import (
    ConvolutionModule as SBConvolutionModule,
)
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import PositionalwiseFeedForward
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm
from torch import nn

try:
    from layers.dss import DSS
except ImportError:
    from .layers.dss import DSS


__all__ = ["S4"]


class S4(nn.Module):
    """S4 model.

    Arguments
    ---------
    input_size : int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
    num_encoder_layers : int, optional
        The number of encoder layers.
    d_ffn : int, optional
        The dimension of the feed forward network model.
    dropout : int, optional
        The dropout value.
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: ReLU or GELU.
    kernel_size : int, optional
        Kernel size in the convolutional layers.
    bias : bool, optional
        Whether to use bias in the convolutional layers.
    causal : bool, optional
        Whether the encoder should be causal.
    s4_layer_type : type, optional
        The S4 layer type.
    s4_layer_kwargs : dict, optional
        The S4 layer keyword arguments.

    Example
    -------
    >>> import torch
    >>>
    >>> batch_size = 4
    >>> seq_length = 2048
    >>> input_size = 80
    >>> d_model = 512
    >>> model = S4(input_size, d_model)
    >>> src = torch.randn(batch_size, seq_length, input_size)
    >>> speaker_embs = torch.randn(batch_size, 1, d_model)
    >>> out = model(src, speaker_embs=speaker_embs)

    """

    def __init__(
        self,
        input_size,
        d_model=512,
        num_encoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.GELU,
        kernel_size: "Optional[int]" = 31,
        bias: "Optional[bool]" = True,
        causal: "Optional[bool]" = False,
        s4_layer_type=DSS,
        s4_layer_kwargs={},
    ):
        super().__init__()

        # Initialize the encoder
        if num_encoder_layers > 0:
            self.encoder = S4Encoder(
                num_layers=num_encoder_layers,
                d_model=d_model,
                d_ffn=d_ffn,
                kernel_size=kernel_size,
                activation=activation,
                bias=bias,
                dropout=dropout,
                causal=causal,
                s4_layer_type=s4_layer_type,
                s4_layer_kwargs=s4_layer_kwargs,
            )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size, n_neurons=d_model, bias=True, combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )

        # Reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, wav_len=None, speaker_embs=None):
        """Forward pass.

        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len : torch.Tensor, optional
            Tensor of shape (batch,) containing the relative
            length to padded length for each example.
        speaker_embs : torch.Tensor, optional
            The speaker embedding.

        Returns
        -------
        torch.Tensor
            The output.

        """
        # Reshape the src vector to [Batch, Time, Features]
        # when a 4D vector is given
        if src.ndim == 4:
            b, t, ch1, ch2 = src.shape
            src = src.reshape(b, t, ch1 * ch2)

        src_key_padding_mask, _ = self._make_masks(src, wav_len)
        src = self.custom_src_module(src)

        src = self.encoder(src=src, src_key_padding_mask=src_key_padding_mask)

        # Inject speaker embedding (at the end to avoid vanishing gradient problems)
        if speaker_embs is not None:
            src += speaker_embs

        return src

    def _make_masks(self, src, wav_len=None):
        if wav_len is not None:
            abs_len = (wav_len * src.shape[1]).round()
            src_key_padding_mask = ~length_to_mask(abs_len).bool()
        else:
            src_key_padding_mask = None
        src_mask = None

        return src_key_padding_mask, src_mask

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


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
    activation : torch.nn.Module
        Activation function used in each S4 layer.
    bias : bool, optional
        Whether bias should be used in the convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal : bool, optional
        Whether the convolutions should be causal or not.
    s4_layer_type : type, optional
        The S4 layer type.
    s4_layer_kwargs : dict, optional
        The S4 layer keyword arguments.

    Example
    -------
    >>> import torch
    >>>
    >>> batch_size = 4
    >>> seq_length = 2048
    >>> d_model = 512
    >>> src = torch.rand((batch_size, seq_length, d_model))
    >>> model = S4Encoder(num_layers=1, d_model=d_model, d_ffn=256)
    >>> out = model(src)

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
        self, src, src_key_padding_mask: "Optional[torch.Tensor]" = None,
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
            output = enc_layer(output, src_key_padding_mask=src_key_padding_mask)
        output = self.norm(output)

        return output


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
    causal : bool, optional
        Whether the convolutions should be causal or not.
    s4_layer_type : type, optional
        The S4 layer type.
    s4_layer_kwargs : dict, optional
        The S4 layer keyword arguments.

    Example
    -------
    >>> import torch
    >>>
    >>> batch_size = 4
    >>> seq_length = 2048
    >>> d_model = 512
    >>> src = torch.rand((batch_size, seq_length, d_model))
    >>> model = S4EncoderLayer(d_model, d_ffn=256)
    >>> out = model(src)

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

        self.s4_layer = s4_layer_type(input_size=d_model, **s4_layer_kwargs)

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


# Fix padding for non-causal
class ConvolutionModule(SBConvolutionModule):
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
        super(SBConvolutionModule, self).__init__()

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


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 2048
    input_size = 80
    d_model = 512
    model = S4(input_size, d_model)
    src = torch.randn(batch_size, seq_length, input_size)
    speaker_embs = torch.randn(batch_size, 1, d_model)
    out = model(src, speaker_embs=speaker_embs)
    print(out.shape)
