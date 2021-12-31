import numpy as np
import pandas as pd
import random, torch, argparse, sys
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, BucketIterator
from tokenization import tokenizer_latex, tokenizer_mml

# set up seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
#only for CUDA
torch.use_deterministic_algorithms(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_args(args):
    parser = argparse.ArgumentParser(description='Preprocess formulas')
    

    parser.add_argument('--input_latex_file', dest='input_latex_file',
                        type=str, required=True,
                        help=('Input file containing latex formulas. One formula per line.'
                        ))

    parser.add_argument('--input_mml_file', dest='input_mml_file',
                        type=str, required=True,
                        help=('Input file containing mml formulas. One formula per line.'
                        ))

    parameters = parser.parse_args(args)
    return parameters

def preprocess(parameters):
    # reading raw text files
    latex_txt = open(parameters.input-latex-file).read().split('\n')
    mml_txt = open(parametres.input-mml-file).read().split('\n')
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
    SRC = Field(tokenize = tokenizer_latex(text), 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                fix_length = 150
                )
                
    TRG = Field(tokenize = tokenizer_mml(text), 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                fix_length = 150
                )
    
    fields = {'Latex': ('latex', SRC) , 'MML': ('mml', TRG)}
    train_data, test_data, val_data = TabularDataset.splits(
          path       = 'data/',
          train_data = 'train.json',
          val_data = 'val.json',
          test_data = 'test.json',
          format     = 'json',
          fields     = fields) 
    
    # Iterator
    train_iter, test_iter, val_iter = BucketIterator(
            (train_data, test_data, val_data), 
            device = devie, 
            batch_size = 256)
    
    # building vocab
    

def main(args):
    parameters = process_args(args)
    preprocess(parameters)
    
if __name__ == '__main__':
    main(sys.argv[1:])
    