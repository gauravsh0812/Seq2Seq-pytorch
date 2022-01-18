# creating tab dataset having image_num and mml
# split train, test, val
# dataloader to load data
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

def preprocess(args_opennmt, device):

    print('preprocessing data...')

    # reading raw text files
    mml_txt = open('data/mml.txt').read().split('\n')
    image_num = list(range(0,len(mml_txt)))
    raw_data = {'IMG': [f'{num}.png' for num in image_num],
                'MML': [mml for mml in mml_txt]}

    df = pd.DataFrame(raw_data, columns=['IMG','MML'])

    train_val, test = train_test_split(df, test_size = 0.1)
    train, val = train_test_split(train_val, test_size=0.1)

    train.to_csv('data/train_i2s.csv', index=False)
    test.to_csv('data/test_i2s.csv', index=False)
    val.to_csv('data/val_i2s.csv', index=False)

    train.to_json('data/train_i2s.json', orient='records', lines=True)
    test.to_json('data/test_i2s.json', orient='records', lines=True)
    val.to_json('data/val_i2s.json', orient='records', lines=True)

    # setting Fields
    # tokenizer will going be default tokenizer i.e. split by spaces
    if args_opennmt ==1:
        TRG = Field(
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    fix_length = 150
                    )

    fields = {'MML': ('mml', TRG)}
    train_data, test_data, val_data = TabularDataset.splits(
          path       = 'data/',
          train = 'train_i2s.json',
          validation = 'val_i2s.json',
          test = 'test_i2s.json',
          format     = 'json',
          fields     = fields)

    # building vocab
    TRG.build_vocab(train_data, min_freq = 10)

    # dataloader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=250)
    return TRG, data_loader
