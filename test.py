# -*- coding: utf-8 -*-

import torch

def evaluate(model, iterator, criterion, args_cnn):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src, src_len = batch.latex
            trg = batch.mml

            if args_cnn == 1:
                # trg = [batch, trg_len]
                trg = trg[:, :-1]
                output = model(src, trg)   # [battch, trg_len-1, output_dim]

            else:
                output = model(src, src_len, trg, True, 0)   # turn off teacher_forcing


            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            if args_cnn == 1:
                output = output.contiguous().view(-1, output_dim)
                trg = trg.contiguous().view(-1)
            else:
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
