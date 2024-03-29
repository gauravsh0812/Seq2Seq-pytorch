import numpy as np
import pandas as pd
import random, torch
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

# set up seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
#only for CUDA
torch.use_deterministic_algorithms(True)

def preprocess(args_cnn, device):

    print('preprocessing data...')

    # reading raw text files
    latex_txt = open('data/latex.txt').read().split('\n')
    mml_txt = open('data/mml.txt').read().split('\n')
    raw_data = {'Latex': [Line for Line in latex_txt],
                'MML': [Line for Line in mml_txt]}

    df = pd.DataFrame(raw_data, columns=['Latex', 'MML'])

    train_val, test = train_test_split(df, test_size = 0.1)
    train, val = train_test_split(train_val, test_size=0.1)

    train.to_csv('data/train_s2s.csv', index=False)
    test.to_csv('data/test_s2s.csv', index=False)
    val.to_csv('data/val_s2s.csv', index=False)

    train.to_json('data/train_s2s.json', orient='records', lines=True)
    test.to_json('data/test_s2s.json', orient='records', lines=True)
    val.to_json('data/val_s2s.json', orient='records', lines=True)

    # setting Fields
    # tokenizer will going be default tokenizer i.e. split by spaces
    # all the src files must be prepared accordingly
    if args_cnn == 0:
        SRC = Field(
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    fix_length = 150,
                    include_lengths = True
                    )

        TRG = Field(
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    fix_length = 150
                    )
    else:
        SRC = Field(
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    fix_length = 150,
                    batch_first = True,
                    include_lengths = True
                    )

        TRG = Field(
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    fix_length = 150,
                    batch_first = True
                    )

    fields = {'Latex': ('latex', SRC) , 'MML': ('mml', TRG)}
    train_data, test_data, val_data = TabularDataset.splits(
          path       = 'data/',
          train = 'train.json',
          validation = 'val.json',
          test = 'test.json',
          format     = 'json',
          fields     = fields)

    # building vocab
    SRC.build_vocab(train_data, min_freq = 10)
    TRG.build_vocab(train_data, min_freq = 10)

    # Iterator
    train_iter, test_iter, val_iter = BucketIterator.splits(
            (train_data, test_data, val_data),
            device = device,
            batch_size = 250,
            sort_within_batch = True,
            sort_key = lambda x: len(x.latex))

    return SRC, TRG, train_iter, test_iter, val_iter
