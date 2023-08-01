"""Transformer model.

Authors
* Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/speechbrain/lobes/models/transformer/TransformerASR.py

from typing import Optional

import torch
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.Transformer import TransformerInterface
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
from torch import nn


__all__ = ["Transformer"]


class Transformer(TransformerInterface):
    """Transformer model.

    Arguments
    ---------
    input_size : int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
    nhead : int, optional
        The number of heads in the multi-head attention models.
    num_encoder_layers : int, optional
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
    encoder_module : str, optional
        Choose between Conformer and Transformer for the encoder.
    conformer_activation : torch.nn.Module, optional
        Activation module used after Conformer convolutional layers e.g. Swish, ReLU etc.
    branchformer_activation : torch.nn.Module, optional
        Activation module used within the Branchformer encoder e.g. Swish, ReLU etc.
    attention_type : str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length : int, optional
        Max length for the source sequence in input.
        Used for positional encodings.
    causal : bool, optional
        Whether the encoder should be causal.
    csgu_linear_units : int, optional
        Number of neurons in the hidden linear units of the CSGU module
        -> Branchformer.
    gate_activation : torch.nn.Module, optional
        Activation function used at the gate of the CSGU module
        -> Branchformer.
    use_linear_after_conv : bool, optional
        If True, will apply a linear transformation of size `input_size // 2`.
        -> Branchformer.

    Example
    -------
    >>> import torch
    >>>
    >>> batch_size = 4
    >>> seq_length = 2048
    >>> input_size = 80
    >>> d_model = 512
    >>> model = Transformer(input_size, d_model)
    >>> src = torch.randn(batch_size, seq_length, input_size)
    >>> speaker_embs = torch.randn(batch_size, 1, d_model)
    >>> out = model(src, speaker_embs=speaker_embs)

    """

    def __init__(
        self,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size: "Optional[int]" = 31,
        bias: "Optional[bool]" = True,
        encoder_module: "Optional[str]" = "conformer",
        conformer_activation: "Optional[nn.Module]" = Swish,
        branchformer_activation: "Optional[nn.Module]" = nn.GELU,
        attention_type: "Optional[str]" = "regularMHA",
        max_length: "Optional[int]" = 2500,
        causal: "Optional[bool]" = False,
        csgu_linear_units: "Optional[int]" = 3072,
        gate_activation: "Optional[nn.Module]" = nn.Identity,
        use_linear_after_conv: "Optional[bool]" = False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
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

        src_key_padding_mask, src_mask = self._make_masks(src, wav_len)

        src = self.custom_src_module(src)
        # Add positional encoding to queries if are sinusoidal
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            pos_embs_encoder = None
            src += self.positional_encoding(src)  # Add the encodings here

        # Inject speaker embedding
        if speaker_embs is not None:
            src *= speaker_embs

        encoder_out, attention_lst = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        return encoder_out

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


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 2048
    input_size = 80
    d_model = 512
    model = Transformer(input_size, d_model)
    src = torch.randn(batch_size, seq_length, input_size)
    speaker_embs = torch.randn(batch_size, 1, d_model)
    out = model(src, speaker_embs=speaker_embs)
    print(out.shape)
