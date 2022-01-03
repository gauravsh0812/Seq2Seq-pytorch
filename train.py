# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder, Decoder, LearningPhrase_Decoder, Seq2Seq
#from preprocessing import preprocess
from preprocessing.preprocessing import preprocess


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

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.latex
        trg = batch.mml
        print(src)
        print(trg)
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
