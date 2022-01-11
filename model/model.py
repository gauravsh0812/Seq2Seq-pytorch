# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import random

class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, hidd_dim, n_layer, dropout):

    super().__init__()

    self.hidd_dim = hidd_dim
    self.n_layer = n_layer
    self.embed = nn.Embedding(input_dim, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hidd_dim, n_layer, dropout=dropout, bidirectional=True)
    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(hidd_dim*2, hidd_dim)

  def forward(self, input):
    # initial hidden, cell =0
    # input -- [input len, batch size]
    # embedded -- [input_len, batch_size, emb_dim]
    embedded = self.drop(self.embed(input))
    # ouput -- [input_len, batch_size, hidd_dim*(n=1/2(if bidirectional))]
    # hidden, cell = [n_layer*n_direction, batch_size, hidd_dim]
    output, (hidden_enc_forward_backward, cell_enc_forward_backward) = self.lstm(embedded)
    hidden = torch.tanh(self.fc(torch.cat((hidden_enc_forward_backward[-2,:,:], hidden_enc_forward_backward[-1,:,:]),dim =1)))
    cell = torch.tanh(self.fc(torch.cat((cell_enc_forward_backward[-2,:,:], cell_enc_forward_backward[-1,:,:]),dim =1)))
    
    return (hidden, cell)

class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, hidd_dim, n_layer, dropout):
    
    super().__init__()
    
    self.hidd_dim = hidd_dim
    self.n_layer = n_layer
    self.output_dim = output_dim
    self.embed = nn.Embedding(output_dim, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hidd_dim, n_layer, dropout=dropout, bidirectional=False)
    #self.fc_hidd = nn.Linear(hidd_dim*2, hidd_dim)
    self.fc_out = nn.Linear(hidd_dim, output_dim) 
    self.drop = nn.Dropout(dropout)

  def forward(self, input, hidden, cell, _):
    # hidden, cell == from the previous encoder
    # input -- [batch size] as input seq length and n_direction will always gonna be one
    # decoder decodes one token at a time only
    # input.unsqeeze() -- [1, batch_size]
    input = input.unsqeeze(0)
    # embedded -- [1, batch_size, emb_dim]
    embedded = self.drop(self.embed(input))
    # ouput -- [1, batch_size, hidd_dim*(n=1/2(if bidirectional))]
    # hidden, cell = [n_layer*n_direction, batch_size, hidd_dim] 
    # Since hidden dim are same, we do not need an additional fc layer
    # but we will goin to need fc layer for enc_output layer. 
    # But now we are not using enc_output.
    output, (hidden_dec_forward_backward, cell_dec_forward_backward) = self.lstm(embedded, hidden.unsqueeze(0), cell.unsqueeze(0)) # hidden, cell as context vector from encoder
    hidden = hidden_dec_forward_backward
    cell = cell_dec_forward_backward
    # prediction -- [batch_size, output_dim]
    prediction = self.fc_out(output.squeeze(0))
    #hidden = torch.tanh(self.fc_hidd(torch.cat((hidden_dec_forward_backward[-2,:,:], hidden_dec_forward_backward[-1,:,:]),dim =1)))
    #cell = torch.tanh(self.fc_hidd(torch.cat((cell_dec_forward_backward[-2,:,:], cell_dec_forward_backward[-1,:,:]),dim =1)))
    return (prediction, hidden, cell)

class LearningPhrase_Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidd_dim, n_layer,  dropout):
        
        super().__init__()
        
        self.hidd_dim = hidd_dim
        self.n_layer = n_layer
        self.output_dim = output_dim
        self.embed = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidd_dim, n_layer, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidd_dim, output_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, context):
        # hidden, cell == from the previous encoder
        # input -- [batch size] as input seq length and n_direction will always gonna be one
        # decoder decodes one token at a time only
        # input.unsqeeze() -- [1, batch_size]
        input = input.unsqeeze(0)
        # embedded -- [1, batch_size, emb_dim]
        embedded = self.drop(self.embed(input))
        # ouput -- [1, batch_size, hidd_dim*(n=1/2(if bidirectional))]
        # context, hidden, cell = [n_layer, batch_size, hidd_dim]
        # context will be added at each step during decoding
        
        embedded_concat = torch.cat((embedded, context), dim=2)
        output, (hidden_lp, cell_lp) = self.lstm(embedded_concat, hidden.unsqueeze(0), cell.unsqueeze(0)) 
        output_concat = torch.cat((output.squueze(0), context.squeeze(0), embedded.squeeze(0)), dim =1)
        hidden = hidden_lp#torch.tanh(self.fc_hidd(torch.cat((hidden_lp[-2,:,:], hidden_lp[-1,:,:]),dim =1)))
        cell = cell_lp#torch.tanh(self.fc_hidd(torch.cat((cell_lp[-2,:,:], cell_lp[-1,:,:]),dim =1)))

        # prediction -- [batch_size, output_dim]
        prediction = self.fc(output_concat)
    
        return (prediction, hidden, cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, learning_phrase=0):
        
        super().__init__()
        
        self.encoder = encoder
        self.device = device
        self.decoder = decoder
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
        context, cell = self.encoder(src)
        hidden = context
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.DEC(input, hidden, cell, context)
            
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

