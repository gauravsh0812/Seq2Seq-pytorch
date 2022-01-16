# -*- coding: utf-8 -*-

import torch

def train(model, iterator, optimizer, criterion, clip, args_attn, args_cnn):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        if args_attn ==1 or args_cnn==1:
            src, src_len = batch.latex
            trg = batch.mml
        optimizer.zero_grad()

        if args_cnn == 1:
            # trg = [batch, trg_len]
            trg = trg[:, :-1]
            output = model(src, trg)   # [battch, trg_len-1, output_dim]

        else:
            output = model(src, src_len, trg, True, 0.5)

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

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
