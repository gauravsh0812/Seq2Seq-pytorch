# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import random

class Encoder(nn.Module):
  def __init__(self, input_dim, hidd_dim, n_layer, emb_dim, dropout):

    super().__init__()

    self.hidd_dim = hidd_dim
    self.n_layer = n_layer
    self.embed = nn.Embedding(input_dim, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hidd_dim, n_layer, dropout=dropout)
    self.drop = nn.Dropout(dropout)

  def forward(self, input):
    # initial hidden, cell =0
    # input -- [input len, batch size]
    # embedded -- [input_len, batch_size, emb_dim]
    embedded = self.drop(self.embed(input))
    # ouput -- [input_len, batch_size, hidd_dim*(n=1/2(if bidirectional))]
    # hidden, cell = [n_layer*n_direction, batch_size, hidd_dim]
    output, (hidden, cell) = self.LSTM(embedded)

    return (hidden, cell)

class Decoder(nn.Module):
  def __init__(self, output_dim, hidd_dim, n_layer, emb_dim, dropout):
    
    super().__init__()
    
    self.hidd_dim = hidd_dim
    self.n_layer = n_layer
    self.embed = nn.Embedding(output_dim, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hidd_dim, n_layer, dropout=dropout)
    self.fc = nn.Linear(hidd_dim, output_dim)
    self.drop = nn.Dropout(dropout)

  def forward(self, input, hidden, cell):
    # hidden, cell == from the previous encoder
    # input -- [batch size] as input seq length and n_direction will always gonna be one
    # decoder decodes one token at a time only
    # input.unsqeeze() -- [1, batch_size]
    # embedded -- [1, batch_size, emb_dim]
    embedded = self.drop(self.embed(input))
    # ouput -- [1, batch_size, hidd_dim*(n=1/2(if bidirectional))]
    # hidden, cell = [n_layer, batch_size, hidd_dim]
    output, (hidden, cell) = self.LSTM(embedded, hidden, cell) # hiiden, cell as context vector from encoder
    # prediction -- [batch_size, output_dim]
    prediction = self.fc(self.hidd_dim, self.output_dim)

    return (prediction, hidden, cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidd_dim == decoder.hidd_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layer == decoder.n_layer, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs

