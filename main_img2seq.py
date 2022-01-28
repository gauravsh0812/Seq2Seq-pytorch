# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from train_im2seq import train
from test_im2seq import evaluate
from preprocessing.preprocessing_img2seq import preprocessing
from preprocessing.preprocessing_images import preprocessing_images
from model.opennmt import Encoder, Decoder, Attention, Seq2Seq
import time
import math
import argparse

# argument
parser = argparse.ArgumentParser()
parser.add_argument( '--opennmt', type=int, metavar='', required=True,
                    help='run OpenNMT replica')
args = parser.parse_args()

def define_model(SRC, TRG, DEVICE):

    #SRC, TRG, train_iter, _, val_iter = preprocess(args_cnn, device)

    print('defining model...')

    INPUT_CHANNEL = SRC.shape[1]
    OUTPUT_DIM = len(TRG.vocab)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    DEC_EMB_DIM = 256
    HID_DIM = 250
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3

    print('building model...')
    enc = Encoder(INPUT_CHANNEL, HID_DIM, ENC_DROPOUT, DEVICE)
    attention = Attention(HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attention)
    model = Seq2Seq(enc, dec, DEVICE)

    return model

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
TRG, data_loader = preprocess(args.opennmt, device)
# train dataset
batch = next(iter(data_loader['train']))
IMG, MML = batch
preprocessed_img = preprocessing_images(IMG, 'data/')
