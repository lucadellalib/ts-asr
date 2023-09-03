"""Models.

Authors
* Luca Della Libera 2023
"""

from typing import List, Optional, Tuple

import torch
import torchaudio
from torchaudio.models import Conformer
from torchaudio.models.rnnt import (
    RNNT,
    _Joiner,
    _Predictor,
    _TimeReduction,
    _Transcriber,
)


__all__ = ["conformer_rnnt_model"]


class _ConformerEncoder(torch.nn.Module, _Transcriber):
    r"""Conformer-based recurrent neural network transducer (RNN-T) encoder (transcription network).

    Args:
        input_dim (int): feature dimension of each input sequence element.
        output_dim (int): feature dimension of each output sequence element.
        time_reduction_input_dim (int, optional): dimension to scale each element in input sequences to prior to applying time reduction block. (Default: 128)
        time_reduction_stride (int, optional): factor by which to reduce length of input sequence. (Default: 2)
        transformer_num_heads (int, optional): number of attention heads in each Conformer layer. (Default: 4)
        transformer_ffn_dim (int, optional): hidden layer dimension of each Conformer layer's feedforward network. (Default: 1024)
        transformer_num_layers (int, optional): number of Conformer layers to instantiate. (Default: 12)
        transformer_depthwise_conv_kernel_size (int, optional): Conformer convolution kernel size. (Default: 31)
        transformer_dropout (float, optional): Conformer dropout probability. (Default: 0.0)
        transformer_use_group_norm (bool, optional): whether to use group norm in the Conformer layers. (Default: False)
        transformer_convolution_first (bool, optional): whether to apply convolution first in the Conformer layers. (Default: False)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        time_reduction_input_dim: int = 128,
        time_reduction_stride: int = 2,
        transformer_num_heads: int = 4,
        transformer_ffn_dim: int = 1024,
        transformer_num_layers: int = 12,
        transformer_depthwise_conv_kernel_size: int = 31,
        transformer_dropout: float = 0.0,
        transformer_use_group_norm: bool = False,
        transformer_convolution_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_linear = torch.nn.Linear(
            input_dim, time_reduction_input_dim, bias=False,
        )
        self.time_reduction = _TimeReduction(time_reduction_stride)
        transformer_input_dim = time_reduction_input_dim * time_reduction_stride
        self.transformer = Conformer(
            input_dim=transformer_input_dim,
            num_heads=transformer_num_heads,
            ffn_dim=transformer_ffn_dim,
            num_layers=transformer_num_layers,
            depthwise_conv_kernel_size=transformer_depthwise_conv_kernel_size,
            dropout=transformer_dropout,
            use_group_norm=transformer_use_group_norm,
            convolution_first=transformer_convolution_first,
        )
        self.output_linear = torch.nn.Linear(transformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(
        self, input: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum input sequence length in batch;
        D: feature dimension of each input sequence frame (input_dim).

        Args:
            input (torch.Tensor): input frame sequences right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output frame sequences.
        """
        input_linear_out = self.input_linear(input)
        time_reduction_out, time_reduction_lengths = self.time_reduction(
            input_linear_out, lengths
        )
        transformer_out, transformer_lengths = self.transformer(
            time_reduction_out, time_reduction_lengths
        )
        output_linear_out = self.output_linear(transformer_out)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, transformer_lengths

    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for inference.

        B: batch size;
        T: maximum input sequence segment length in batch;
        D: feature dimension of each input sequence frame (input_dim).

        Args:
            input (torch.Tensor): input frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``infer``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation
                    of ``infer``.
        """
        input_linear_out = self.input_linear(input)
        time_reduction_out, time_reduction_lengths = self.time_reduction(
            input_linear_out, lengths
        )
        (
            transformer_out,
            transformer_lengths,
            transformer_states,
        ) = self.transformer.infer(time_reduction_out, time_reduction_lengths, states)
        output_linear_out = self.output_linear(transformer_out)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, transformer_lengths, transformer_states


def conformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    num_symbols: int,
    time_reduction_input_dim: int = 128,
    time_reduction_stride: int = 2,
    transformer_num_heads: int = 4,
    transformer_ffn_dim: int = 1024,
    transformer_num_layers: int = 12,
    transformer_depthwise_conv_kernel_size: int = 31,
    transformer_dropout: float = 0.0,
    transformer_use_group_norm: bool = False,
    transformer_convolution_first: bool = False,
    symbol_embedding_dim: int = 256,
    num_lstm_layers: int = 1,
    lstm_layer_norm: bool = False,
    lstm_layer_norm_epsilon: float = 1e-6,
    lstm_dropout: float = 0.0,
) -> torchaudio.models.RNNT:
    r"""Builds Conformer-based :class:`~torchaudio.models.RNNT`.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        num_symbols (int): cardinality of set of target tokens.
        time_reduction_input_dim (int, optional): dimension to scale each element in input sequences to prior to applying time reduction block. (Default: 128)
        time_reduction_stride (int, optional): factor by which to reduce length of input sequence. (Default: 2)
        transformer_num_heads (int, optional): number of attention heads in each Conformer layer. (Default: 4)
        transformer_ffn_dim (int, optional): hidden layer dimension of each Conformer layer's feedforward network. (Default: 1024)
        transformer_num_layers (int, optional): number of Conformer layers to instantiate. (Default: 12)
        transformer_depthwise_conv_kernel_size (int, optional): Conformer convolution kernel size. (Default: 31)
        transformer_dropout (float, optional): Conformer dropout probability. (Default: 0.0)
        transformer_use_group_norm (bool, optional): whether to use group norm in the Conformer layers. (Default: False)
        transformer_convolution_first (bool, optional): whether to apply convolution first in the Conformer layers. (Default: False)
        symbol_embedding_dim (int, optional): dimension of each target token embedding. (Default: 256)
        num_lstm_layers (int, optional): number of LSTM layers to instantiate. (Default: 1)
        lstm_layer_norm (bool, optional): if ``True``, enables layer normalization for LSTM layers. (Default: False)
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in LSTM layer normalization layers. (Default: 1e-6)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)

    Returns:
        RNNT:
            Conformer RNN-T model.
    """
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_input_dim=time_reduction_input_dim,
        time_reduction_stride=time_reduction_stride,
        transformer_num_heads=transformer_num_heads,
        transformer_ffn_dim=transformer_ffn_dim,
        transformer_num_layers=transformer_num_layers,
        transformer_depthwise_conv_kernel_size=transformer_depthwise_conv_kernel_size,
        transformer_dropout=transformer_dropout,
        transformer_use_group_norm=transformer_use_group_norm,
        transformer_convolution_first=transformer_convolution_first,
    )
    predictor = _Predictor(
        num_symbols,
        encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=symbol_embedding_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols)
    return RNNT(encoder, predictor, joiner)
