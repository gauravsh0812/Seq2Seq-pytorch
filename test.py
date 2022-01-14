# -*- coding: utf-8 -*-

import torch
#import torch.nn as nn
#import torch.optim as optim
#from model.model import Encoder, Decoder, LearningPhrase_Decoder, Seq2Seq
#from preprocessing.preprocessing import preprocess

def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.latex
            trg = batch.mml

            if args_cnn == 1:
                # trg = [batch, trg_len]
                trg = trg[:, :-1]
                output = model(src, trg)   # [battch, trg_len-1, output_dim]

            else:
                output = model(src, trg, True, 0)   # turn off teacher_forcing


            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            if args_cnn == 1:
                output = output.view(-1, output_dim)
                trg = trg[:, 1:].veiw(-1)
            else:
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
