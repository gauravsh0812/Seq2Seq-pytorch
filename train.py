# -*- coding: utf-8 -*-

import torch

def train(model, iterator, optimizer, criterion, clip):

    print('Let\'s start training!')

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.latex
        trg = batch.mml
        optimizer.zero_grad()

        output = model(src, trg, 0.5, True)

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
