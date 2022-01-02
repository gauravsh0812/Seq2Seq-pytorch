import numpy as np
import pandas as pd
import random, torch, argparse, sys
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess():
    
    # reading raw text files
    latex_txt = open('data/latex.txt').read().split('\n')
    mml_txt = open('data/mml.txt').read().split('\n')
    raw_data = {'Latex': [Line for Line in latex_txt],
                'MML': [Line for Line in mml_txt]}
    
    df = pd.DataFrame(raw_data, columns=['Latex', 'MML'])
    
    train_val, test = train_test_split(df, test_size = 0.1)
    train, val = train_test_split(train_val, test_size=0.1)
    
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    
    train.to_json('data/train.json', orient='records', lines=True)
    test.to_json('data/test.json', orient='records', lines=True)
    val.to_json('data/val.json', orient='records', lines=True)
     
    # setting Fields
    SRC = Field( 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                fix_length = 150
                )
                
    TRG = Field( 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                fix_length = 150
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
            batch_size = 256)
    
    return SRC, TRG, train_iter, test_iter, val_iter
    
