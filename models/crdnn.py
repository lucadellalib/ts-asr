"""CRDNN model.

Authors
* Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.15/speechbrain/lobes/models/CRDNN/CRDNN.py

import torch
from speechbrain.lobes.models.CRDNN import CRDNN as SBCRDNN


__all__ = ["CRDNN"]


class CRDNN(SBCRDNN):
    """This model is a combination of CNNs, RNNs, and DNNs.

    This model expects 3-dimensional input [batch, time, feats] and
    by default produces output of the size [batch, time, dnn_neurons].

    One exception is if ``using_2d_pooling`` or ``time_pooling`` is True.
    In this case, the time dimension will be downsampled.

    Arguments
    ---------
    input_size : int
        The length of the expected input at the third dimension.
    input_shape : tuple
        While input_size will suffice, this option can allow putting
        CRDNN into a sequential with other classes.
    activation : torch class
        A class used for constructing the activation layers for CNN and DNN.
    dropout : float
        Neuron dropout rate as applied to CNN, RNN, and DNN.
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_channels : list of ints
        A list of the number of output channels for each CNN block.
    cnn_kernelsize : tuple of ints
        The size of the convolutional kernels.
    time_pooling : bool
        Whether to pool the utterance on the time axis before the RNN.
    time_pooling_size : int
        The number of elements to pool on the time axis.
    time_pooling_stride : int
        The number of elements to increment by when iterating the time axis.
    using_2d_pooling : bool
        Whether using a 2D or 1D pooling after each CNN block.
    inter_layer_pooling_size : list of ints
        A list of the pooling sizes for each CNN block.
    rnn_class : torch class
        The type of RNN to use in CRDNN network (LiGRU, LSTM, GRU, RNN).
    rnn_layers : int
        The number of recurrent RNN layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_bidirectional : bool
        Whether this model will process just forward or in both directions.
    rnn_re_init : bool,
        If True, an orthogonal initialization will be applied to the recurrent
        weights.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.
    use_rnnp : bool
        If True, a linear projection layer is added between RNN layers.
    projection_dim : int
        The number of neurons in the projection layer.
        This layer is used to reduce the size of the flattened
        representation obtained after the CNN blocks.

    Example
    -------
    >>> import torch
    >>>
    >>> batch_size = 4
    >>> seq_length = 2048
    >>> input_size = 80
    >>> cnn_channels = [128, 256]
    >>> model = CRDNN(input_size, cnn_channels=cnn_channels)
    >>> src = torch.randn(batch_size, seq_length, input_size)
    >>> speaker_embs = torch.randn(batch_size, 1, cnn_channels[-1])
    >>> out = model(src, speaker_embs=speaker_embs)

    """

    def forward(self, x, wav_len=None, speaker_embs=None):
        """Applies layers in sequence, passing only the first element of tuples.
        The speaker embedding is injected before the first RNN layer.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to forward through the network.
        wav_len : torch.Tensor, optional
            Unused, kept for compatibility reasons.
        speaker_embs : torch.Tensor, optional
            The speaker embedding.

        """
        for layer in self.values():
            # Inject speaker embedding before first RNN layer
            if speaker_embs is not None:
                if hasattr(layer, "rnn"):
                    x += speaker_embs[..., None, :]
                    speaker_embs = None

            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

        return x


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 2048
    input_size = 80
    cnn_channels = [128, 256]
    model = CRDNN(input_size, cnn_channels=cnn_channels)
    src = torch.randn(batch_size, seq_length, input_size)
    speaker_embs = torch.randn(batch_size, 1, cnn_channels[-1])
    out = model(src, speaker_embs=speaker_embs)
    print(out.shape)
