#!/usr/bin/env bash

# NOTE: datasets/LibriSpeech must exist

python train_librispeech.py hparams/LibriSpeech/rnn-t.yaml --data_folder datasets/LibriSpeech \
--rnn_neurons 100 --rnn_layers 1 --decoder_neurons 100 --joint_dim 100 \
--train_batch_size 2 --valid_batch_size 1 --beam_size 1  --run_pretrainer False \
--debug

python train_librispeech.py hparams/LibriSpeech/conformer-t.yaml --data_folder datasets/LibriSpeech \
--d_model 80 --d_ffn 100 --num_encoder_layers 1 --decoder_neurons 100 --joint_dim 100 \
--train_batch_size 2 --valid_batch_size 1 --beam_size 1 --run_pretrainer False \
--debug

python train_librispeech.py hparams/LibriSpeech/s4-t.yaml --data_folder datasets/LibriSpeech \
--d_model 80 --d_ffn 100 --num_encoder_layers 1 --decoder_neurons 100 --joint_dim 100 \
--train_batch_size 2 --valid_batch_size 1 --beam_size 1 --run_pretrainer False \
--debug
