"""Conformer model.

Authors
* Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/speechbrain/lobes/models/transformer/Conformer.py
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/speechbrain/lobes/models/transformer/TransformerASR.py

from typing import List, Optional, Union

import torch
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoderLayer
from speechbrain.lobes.models.transformer.Transformer import (
    PositionalEncoding,
    RelPosEncXL,
    get_lookahead_mask,
)
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import MultiheadAttention
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
from torch import nn


__all__ = ["ConformerEncoder"]


class ConformerEncoder(nn.Module):
    """Conformer encoder model.

    Arguments
    ---------
    input_size : int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
    nhead : int, optional
        The number of heads in the multi-head attention models.
    num_layers : int, optional
        The number of encoder layers.
    d_ffn : int, optional
        The dimension of the feed forward network model.
    dropout : int, optional
        The dropout value.
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: ReLU or GELU.
    positional_encoding : str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine'
        for fixed absolute positional encodings.
    normalize_before : bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size : int, optional
        Kernel size in the convolutional layers when Conformer is used.
    bias : bool, optional
        Whether to use bias in Conformer convolutional layers.
    conformer_activation : torch.nn.Module, optional
        Activation module used after Conformer convolutional layers e.g. Swish, ReLU etc.
    attention_type : str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length : int, optional
        Max length for the source sequence in input.
        Used for positional encodings.
    causal : bool, optional
        Whether the encoder should be causal.
    injection_mode : str, optional
        The embedding injection mode (prod, sum, cat, cross_attention, or None).

    Example
    -------
    >>> import torch
    >>>
    >>> batch_size = 4
    >>> seq_length = 256
    >>> input_size = 80
    >>> d_model = 512
    >>> model = ConformerEncoder(input_size, d_model)
    >>> src = torch.randn(batch_size, seq_length, input_size)
    >>> speaker_embs = torch.randn(batch_size, 1, d_model)
    >>> out = model(src, speaker_embs=speaker_embs)

    """

    def __init__(
        self,
        input_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        d_ffn=2048,
        dropout=0.0,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size: "Optional[int]" = 31,
        bias: "Optional[bool]" = True,
        conformer_activation: "Optional[nn.Module]" = Swish,
        attention_type: "Optional[str]" = "regularMHA",
        max_length: "Optional[int]" = 2500,
        causal: "Optional[bool]" = False,
        injection_mode: "Optional[str]" = "prod",
        injection_after: "Union[int, List[int]]" = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.activation = activation
        self.positional_encoding_type = positional_encoding_type = positional_encoding
        self.normalize_before = normalize_before
        self.kernel_size = kernel_size
        self.bias = bias
        self.conformer_activation = conformer_activation
        self.attention_type = attention_type
        self.max_length = max_length
        self.causal = causal
        self.injection_mode = injection_mode
        self.injection_after = injection_after
        if not isinstance(injection_after, (list, tuple)):
            self.injection_after = [injection_after]

        if positional_encoding_type == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(d_model, max_length)

        # Overrides any other pos_embedding
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size, n_neurons=d_model, bias=True, combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

        if injection_mode == "cat":
            self.cat_proj = Linear(
                input_size=2 * d_model, n_neurons=d_model, bias=True,
            )
        elif injection_mode == "cross_attention":
            self.speaker_attn = MultiheadAttention(nhead, d_model, dropout, bias)

        # Reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, wav_len=None, speaker_embs=None, speaker_embs_length=None):
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
        speaker_embs_length : torch.Tensor, optional
            The speaker embedding length (used only if `injection_mode` == "cross_attention").

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

        src_key_padding_mask, src_mask = self._make_masks(src, wav_len)
        src = self.custom_src_module(src)

        # Add positional encoding to queries if are sinusoidal
        if self.attention_type == "RelPosMHAXL":
            pos_embs = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            pos_embs = None
            src += self.positional_encoding(src)  # Add the encodings here

        for i, layer in enumerate(self.layers):
            src, _ = layer(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )

            # Inject speaker embedding
            if i in self.injection_after:
                if speaker_embs is not None:
                    if self.injection_mode == "prod":
                        src *= speaker_embs
                    elif self.injection_mode == "sum":
                        src += speaker_embs
                    elif self.injection_mode == "cat":
                        src = torch.cat(
                            [src, speaker_embs.expand(-1, src.shape[-2], -1)], dim=-1
                        )
                        src = self.cat_proj(src)
                    elif self.injection_mode == "cross_attention":
                        key_padding_mask = None
                        if speaker_embs_length is not None:
                            key_padding_mask = ~length_to_mask(
                                (speaker_embs_length * speaker_embs.shape[-2]).round()
                            ).bool()
                        src, _ = self.speaker_attn(
                            src,
                            speaker_embs,
                            speaker_embs,
                            key_padding_mask=key_padding_mask,
                        )
                    elif self.injection_mode is None:
                        pass
                    else:
                        raise NotImplementedError

        src = self.norm(src)
        return src

    def _make_masks(self, src, wav_len=None):
        if wav_len is not None:
            abs_len = (wav_len * src.shape[1]).round()
            src_key_padding_mask = ~length_to_mask(abs_len).bool()
        else:
            src_key_padding_mask = None

        src_mask = None

        if self.causal:
            src_mask = get_lookahead_mask(src)

        return src_key_padding_mask, src_mask

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 256
    input_size = 80
    d_model = 512
    model = ConformerEncoder(input_size, d_model)
    src = torch.randn(batch_size, seq_length, input_size)
    speaker_embs = torch.randn(batch_size, 1, d_model)
    out = model(src, speaker_embs=speaker_embs)
    print(out.shape)