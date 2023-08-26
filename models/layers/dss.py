"""Simplified diagonal state space (DSS) layer.

Authors
* Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/lucidrains/gated-state-spaces-pytorch/blob/v0.0.16/gated_state_spaces_pytorch/gss.py

import torch
from torch import nn
from torch.fft import irfft, rfft


__all__ = ["DSS"]


class DSS(nn.Module):
    """Simplified diagonal state space (DSS) layer.

    Arguments
    ---------
    input_size:
        The input size (i.e. the number of features).
    state_size:
        The state size.

    References
    ----------
    .. [1] H. Mehta, A. Gupta, A. Cutkosky, and B. Neyshabur.
           "Long Range Language Modeling via Gated State Spaces".
           In: International Conference on Learning Representations (ICLR). 2023.
           URL: https://arxiv.org/abs/2206.13947v3

    Examples
    --------
    >>> import torch
    >>>
    >>> batch_size = 4
    >>> seq_length = 256
    >>> input_size = 64
    >>> model = DSS(input_size)
    >>> input = torch.randn(batch_size, seq_length, input_size)
    >>> output = model(input)

    """

    def __init__(self, input_size: "int", state_size: "int" = 512,) -> "None":  # H  # N
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size

        # Input normalization
        self.norm = nn.LayerNorm(input_size)

        # Lambda
        self.Lambda_real = nn.Parameter(torch.randn(state_size))
        self.Lambda_imag = nn.Parameter(torch.randn(state_size))

        # C
        self.C_real = nn.Parameter(torch.randn(input_size, state_size))
        self.C_imag = nn.Parameter(torch.randn(input_size, state_size))

        # D
        self.D = nn.Parameter(torch.randn(input_size))

    def _build_kernel(self, seq_length: "int") -> "torch.Tensor":
        # [state_size]
        Lambda_real = -self.Lambda_real.exp()
        Lambda_imag = self.Lambda_imag.exp()
        Lambda = Lambda_real + 1j * Lambda_imag

        # [input_size, state_size]
        C = (self.C_real + 1j * self.C_imag) * (Lambda.exp() - 1) / Lambda

        range = torch.arange(seq_length, device=C.device)
        # [state_size, seq_length]
        S = (Lambda[:, None] * range[None, :]).exp()

        # [input_size, seq_length]
        K = (C @ S).real

        return K

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Arguments
        ---------
        input:
            The input, shape: ``[*batch_shape, seq_length, input_size]``.

        Returns
        -------
            The output, shape: ``[*batch_shape, seq_length, input_size]``.

        """
        u = input

        # Build kernel
        seq_length = u.shape[-2]  # L
        # [input_size, seq_length]
        K = self._build_kernel(seq_length)

        # [*batch_shape, seq_length, input_size]
        u = self.norm(u)

        # Learned weighted residual
        # [*batch_shape, seq_length, input_size]
        residual = u * self.D

        # Conv1D FFT (nlog(n))
        fft_length = 2 * seq_length
        # [*batch_shape, seq_length + 1, input_size]
        u_f = rfft(u, n=fft_length, dim=-2)
        # [input_size, seq_length + 1]
        K_f = rfft(K.T, n=fft_length, dim=-2)
        # [*batch_shape, seq_length, input_size]
        y = irfft(u_f * K_f, n=fft_length, dim=-2)[..., :seq_length, :]
        output = y + residual

        return output


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 256
    input_size = 64
    model = DSS(input_size)
    input = torch.randn(batch_size, seq_length, input_size)
    output = model(input)
    print(output.shape)
