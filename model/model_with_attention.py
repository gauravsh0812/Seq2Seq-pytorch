# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hidd_dim, dec_hidd_dim, n_layer, dropout):
        
        super().__init__()
        self.enc_hidd_dim = enc_hidd_dim
        self.dec_hidd_dim = dec_hidd_dim
        self.n_layer = n_layer
        
        self.embed = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, enc_hidd_dim, n_layer, dropout=dropout, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidd_dim*2, dec_hidd_dim)
    
    def forward(self, input):
        # input = [src_len, batch]
        embedded = self.drop(self.embed(input))
        # embedded = [src_len, batch, emb_dim]
        output, (hidden_forward_backward, cell_forward_backward) = self.lstm(embedded)
        # output = [src_len, batch,  enc_hidd_dim*n_directions]
        # hidden_for_back = cell_for_back = [n_layer*n_directions, batch, enc_hidd_dim]
        forward_hidden, backward_hidden = hidden_forward_backward[-2,:,:], hidden_forward_backward[-1,:,:]
        # backward/forward_hidden = [1,:,:]
        forward_cell, backward_cell = cell_forward_backward[-2,:,:], cell_forward_backward[-1,:,:]
        hidden = torch.tanh(self.fc(torch.cat((forward_hidden, backward_hidden),dim=1)))
        # concat_hidden/cell =[1,:,enc_hidd_dim*2]
        # after fc = [1,:,dec_hidd_dim]
        cell = torch.tanh(self.fc(torch.cat((forward_cell, backward_cell),dim=1)))
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidd_dim, dec_hidd_dim, n_layer, dropout):
        
        super().__init__()
        self.enc_hidd_dim = enc_hidd_dim
        self.dec_hidd_dim = dec_hidd_dim
        self.n_layer = n_layer
        
        self.embed = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, dec_hidd_dim, n_layer, dropout=dropout, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidd_dim*2, dec_hidd_dim)
        
            