#!/usr/bin/env bash

# NOTE: datasets/LibriSpeechMix must exist

python train_librispeech_mix.py hparams/LibriSpeechMix/rnn-t.yaml --data_folder datasets/LibriSpeechMix \
--rnn_neurons 100 --rnn_layers 1 --decoder_neurons 100 --joint_dim 100 \
--max_batch_size 2 --beam_size 1 --run_pretrainer False \
--debug

python train_librispeech_mix.py hparams/LibriSpeechMix/conformer-t.yaml --data_folder datasets/LibriSpeechMix \
--d_model 80 --d_ffn 100 --num_encoder_layers 1 --decoder_neurons 100 --joint_dim 100 \
--max_batch_size 2 --beam_size 1 --run_pretrainer False \
--debug

python train_librispeech_mix.py hparams/LibriSpeechMix/s4-t.yaml --data_folder datasets/LibriSpeechMix \
--d_model 80 --d_ffn 100 --num_encoder_layers 1 --decoder_neurons 100 --joint_dim 100 \
--max_batch_size 2 --beam_size 1 --run_pretrainer False \
--debug
