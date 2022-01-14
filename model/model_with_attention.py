# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class Encoder_Attn(nn.Module):
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
        # concat_hidden/cell =[1,batch,enc_hidd_dim*2]
        # after fc = [1,batch,dec_hidd_dim]
        cell = torch.tanh(self.fc(torch.cat((forward_cell, backward_cell),dim=1)))

        return output, hidden, cell

class Attention(nn.Module):
    # Energy_t = tanh(attn(s_t-1, all enc_outputs))
    # a_t = v*Energy_t

        def __init__(self, enc_hidd_dim, dec_hidd_dim):
            self.attn = torch.tanh(nn.Linear(enc_hidd_dim*2 + dec_hidd_dim, dec_hidd_dim))
            self.v = nn.Linear(dec_hidd_dim, 1, bias = False)

        def forward(self, hidden, enc_outputs):
            # hidden = [batch, dec_hidd_dim]
            # enc_outputs = [seq_len, batch, enc_hidd_dim]
            src_len = enc_outputs.shape[0]

            hidden = hidden.unsqueeze(1).repeat(src_len, 1, 1)
            # hidden = [src_len, batch, dec_hidd_dim]

            energy = self.attn(torch.cat(hidden, enc_outputs), dim=2)
            # torch.cat(hidden, enc_outputs), dim=2 ==> [src_len, batch, enc_hidd_dim*2+dec_hidd_dim]
            # energy = [src_len, batch, dec_hidd_dim] and v=[dec_hidd_dim, 1]
            attn_vector = self.v(energy).squeeze(2)   # [src_len, batch]
            attn_vector = attn_vector.permute(1,0)    # [batch, src_len]

            return F.softmax(attn_vector, dim=1)

class Decoder_Attn(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidd_dim, dec_hidd_dim, n_layer, dropout, attention):

        super().__init__()

        self.enc_hidd_dim = enc_hidd_dim
        self.dec_hidd_dim = dec_hidd_dim
        self.n_layer = n_layer
        self.attention = attention
        self.output_dim = output_dim

        self.embed = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, dec_hidd_dim, n_layer, dropout=dropout, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidd_dim*2, dec_hidd_dim)

    def forward(self, input, hidden, cell, encoder_outputs):

        # wgt = a_t * H
        input = input.unsqueeze(0)
        embedded = self.drop(self.embed(input))   # [1, batch, emb_dim]
        a_t = self.attention(hidden, encoder_outputs).unsqueeze(1) # [batch, 1, src_len]
        # since in decoder we don't need src_len as it will be equals to 1
        encoder_outputs = encoder_outputs.permute(1,0,2)   #[batch, src_len, enc_hidd_dim*2]
        wgt = torch.bmm(a_t, encoder_outputs)
        # bmm is batch matrix multioplication therefore, [1,src_len] * {src_len, enc_hidd_dim*2] ==> [1, enc_...]
        # wgt = [batch, 1, enc_...]
        lstm_input = torch.cat((wgt.permute(1,0,2), embedded), dim=2)  # [1, batch, enc_hidd_dim*2 + emb_dim]
        output, hidden, cell = self.lstm(lstm_input, hidden.unsqueeze(0), cell.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        wgt = wgt.squeeze(0)

        prediction = self.fc_out(torch.cat((output, wgt, embedded), dim = 1))

        #prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)

class Seq2Seq_Attn(nn.Module):
    def __init__(self, encoder, decoder, attention, device):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device

    def forward(self, src, trg, teacher_force_flag, teacher_forcing_ratio=0.5):
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
        encoder_outputs, hidden, cell = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #decide if we are going to use teacher forcing or not
            teacher_force = False
            if teacher_force_flag:
                teacher_force = random.random() < teacher_forcing_ratio

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
