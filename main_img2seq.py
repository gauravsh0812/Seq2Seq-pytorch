# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from train import train
from test import evaluate
from preprocessing.preprocessing import preprocess
from model.model import Encoder, Decoder, LearningPhrase_Decoder, Seq2Seq
from model.model_with_attention import Encoder_Attn, Decoder_Attn, Seq2Seq_Attn, Attention
from model.CNN_seq2seq_model import Encoder_CNN, Decoder_CNN, Seq2Seq_CNN
import time
import math
import argparse

# argument
parser = argparse.ArgumentParser()
parser.add_argument( '--opennmt', type=int, metavar='', required=True,
                    help='run OpenNMT replica')
args = parser.parse_args()

def define_model(args_learning_phrase, args_attn, args_cnn, SRC, TRG, device):

    #SRC, TRG, train_iter, _, val_iter = preprocess(args_cnn, device)

    print('defining model...')

    SRC_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    N_LAYERS = 1
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    if args_cnn == 1:
        ENC_LAYERS = 10
        DEC_LAYERS = 10
        ENC_KERNEL_SIZE = 3
        DEC_KERNEL_SIZE = 3

    print(args_cnn)
    print(args_attn)

    if args_cnn == 1:
        print('in CNN model')
        enc = Encoder_CNN(SRC_DIM, ENC_EMB_DIM, ENC_HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT,  device)
        dec = Decoder_CNN(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

        model = Seq2Seq_CNN(enc, dec)

    elif args_attn == 1:
        print('in attn model')
        attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder_Attn(src_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder_Attn(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT, attention)

        model = Seq2Seq_Attn(enc, dec, SRC_PAD_IDX, device)

    else:

        enc = Encoder(src_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)
        lp_dec = LearningPhrase_Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)

        DEC = dec if args_learning_phrase == 0 else lp_dec

        model = Seq2Seq(enc, DEC, device, args_learning_phrase)

    return model
