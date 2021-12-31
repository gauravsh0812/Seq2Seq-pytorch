import os, shutil, subprocess, sys, argparse, logging
from mml_cleaner import eliminate_keywords, reduce_mml

def process_args(args, latex_flag=True):
    parser = argparse.ArgumentParser(description='Preprocess formulas')
    
    if latex_flag:
        parser.add_argument('--input-latex-file', dest='input_latex_file',
                            type=str, required=True,
                            help=('Input file containing latex formulas. One formula per line.'
                            ))
        parser.add_argument('--output-latex-file', dest='output_latex_file',
                            type=str, required=True,
                            help=('Output file.'
                            ))
    else:
        parser.add_argument('--input-mml-file', dest='input_mml_file',
                            type=str, required=True,
                            help=('Input file containing mml formulas. One formula per line.'
                            ))
        parser.add_argument('--output-mml-file', dest='output_mml_file',
                            type=str, required=True,
                            help=('Output file.'
                            ))
        
    parameters = parser.parse_args(args)
    return parameters


def is_ascii(str):
    try:
        str.decode('ascii')
        return True
    except UnicodeError:
        return False

def tokenizer_latex(latex_args):
    
    parameters = process_args(latex_args)
        
    input_file = parameters.input_file
    output_file = parameters.output_file

    assert os.path.exists(input_file), input_file
    cmd = "perl -pe 's|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g' %s > %s"%(input_file, output_file)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logging.error('FAILED: %s'%cmd)

    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as fout:  
        with open(output_file) as fin:
            for line in fin:
                fout.write(line.replace('\r', ' ').strip() + '\n')  # delete \r

    cmd = "cat %s | node ../preprocessing/pp_latex_node.js tokenize > %s "%(temp_file, output_file)
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    if ret != 0:
        logging.error('FAILED: %s'%cmd)
    temp_file = output_file + '.tmp'
    shutil.move(output_file, temp_file)
    with open(temp_file) as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    if is_ascii(token):
                        tokens_out.append(token)
                fout.write(' '.join(tokens_out)+'\n')
    os.remove(temp_file)

def tokenizer_mml(mml_args):
    
    parameters = process_args(mml_args,latex_flag=False)
        
    input_file = parameters.input_file
    output_file = parameters.output_file
    temp_list = eliminate_keywords(input_file)
    reduced_temp_list = reduce_mml(temp_list)
    

def main(args):
    tokenizer_latex(args[0:2])
    tokenizer_mml(args[2:])
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    