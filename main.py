# -*- coding: utf-8 -*-

import torch
from train import train, define_model, optimizer, criterion
from test import evaluate 
from preprocessing.preprocessing import preprocess
import time
import math

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

_, _, train_iter, test_iter, val_iter = preprocess()

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
