# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from train import train
from test import evaluate
from preprocessing.preprocessing import preprocess
from model.model import Encoder, Decoder, LearningPhrase_Decoder, Seq2Seq
from model.model_with_attention import Encoder_Attn, Decoder_Att, Seq2Seq_Attn, Attention
from model.CNN_seq2seq_model import Encoder_CNN, Decoder_CNN, Seq2Seq_CNN
import time
import math
import argparse

# argument
parser = argparse.ArgumentParser()
parser.add_argument( '--learning_phrase', type=int, metavar='', required=True,
                    help='Learning Phrase Decoder')
parser.add_argument( '--Seq2Seq_with_attention', type=int, metavar='', required=True,
                    help='run model with attention')
parser.add_argument( '--CNN_Seq2Seq_with_attention', type=int, metavar='', required=True,
                    help='use CNN2CNN for Seq2Seq')
args = parser.parse_args()


def define_model(args_learning_phrase, args_attn, args_cnn, SRC, TRG, device):

    #SRC, TRG, train_iter, _, val_iter = preprocess(args_cnn, device)

    print('defining model...')

    src_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
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


    if args_cnn == 1:
        enc = Encoder_CNN(SRC_DIM, ENC_EMB_DIM, ENC_HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT,  device)
        dec = Decoder_CNN(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

        model = Seq2Seq_CNN(enc, dec)

    elif args_attn == 1:
        attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder_Attn(src_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder_Attn(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT, attention)

        model = Seq2Seq_Attn(enc, dec, SRC_PAD_IDX device)

    else:

        enc = Encoder(src_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)
        lp_dec = LearningPhrase_Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)

        DEC = dec if args_learning_phrase == 0 else lp_dec

        model = Seq2Seq(enc, DEC, device, args_learning_phrase)

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


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

SRC, TRG, train_iter, test_iter, val_iter = preprocess(args.CNN, device)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = define_model(args.learning_phrase, args.Seq2Seq_with_attention,
                          args.CNN_Seq2Seq_with_attention, SRC, TRG, device)
model.to(device)

print('MODEL: ')
print(model.apply(init_weights))
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

if args.Seq2Seq_with_attention ==1: model_name = 'seq2seq_attn'
elif args.CNN_Seq2Seq_with_attention ==1:model_name = 'CNN_seq2seq'
elif args.learning_phrase==1: model_name = 'seq2seq_lpr'
else: model_name = 'seq2seq'

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'{model_name}-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load(f'{model_name}-model.pt'))

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
