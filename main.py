# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from train import train
from test import evaluate 
from preprocessing.preprocessing import preprocess
from model.model import Encoder, Decoder, LearningPhrase_Decoder, Seq2Seq
import time
import math
import argparse

# argument
parser = argparse.ArgumentParser()
parser.add_argument( '--learning_phrase', type=int, metavar='', required=True, 
                    help='Learning Phrase Decoder')
args = parser.parse_args()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

_, _, train_iter, test_iter, val_iter = preprocess()

def define_model(learning_phrase=0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    SRC, TRG, train_iter, _, val_iter = preprocess()
    
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    lp_dec = LearningPhrase_Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, lp_dec, learning_phrase, device).to(device)
    
    return model, TRG

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
define_model()[0].apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(define_model()[0]):,} trainable parameters')

optimizer = optim.Adam(define_model()[0].parameters())

TRG_PAD_IDX = define_model()[1].vocab.stoi[define_model()[1].pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(define_model()[0], train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(define_model()[0], val_iter, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(define_model()[0].state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


define_model()[0].load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(define_model()[0], test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
