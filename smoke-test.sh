#!/usr/bin/env bash

# NOTE: datasets/LibriSpeechMix must exist

python train.py hparams/rnn-t.yaml --data_folder datasets/LibriSpeechMix \
--sorting ascending --rnn_neurons 100 --rnn_layers 1 --decoder_neurons 100 --joint_dim 100 --train_batch_size 2 --valid_batch_size 1 --beam_size 1 \
--debug

python train.py hparams/conformer-t.yaml --data_folder datasets/LibriSpeechMix \
--sorting ascending --d_model 100 --d_ffn 100 --num_encoder_layers 1 --decoder_neurons 100 --joint_dim 100 --train_batch_size 2 --valid_batch_size 1 --beam_size 1 \
--debug

python train.py hparams/s4-t.yaml --data_folder datasets/LibriSpeechMix \
--sorting ascending --d_model 100 --d_ffn 100 --num_encoder_layers 1 --decoder_neurons 100 --joint_dim 100 --train_batch_size 2 --valid_batch_size 1 --beam_size 1 \
--debug
