import re

path_to_mml = 'data/modified_mml.txt'
path_to_tok_mml = 'data/mml.txt'
mml_txt = open(path_to_mml, 'r').readlines()
tokenized_mml = open(path_to_tok_mml, 'w')

res=' '

for idx, m in enumerate(mml_txt):
    if idx%100==0: print(idx)
    mml_split = re.split('>|<',m)
    for token in mml_split:
        token = token.strip()
        if len(token)>0:
            if '&#x' in  token or len(token)==1 or token.isdecimal():
                res += token
            else:
                res += ' <' + token +'> '
    tokenized_mml.write(res)
