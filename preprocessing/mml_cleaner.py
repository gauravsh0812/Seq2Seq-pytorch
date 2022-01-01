import os

def eliminate_keywords(mml_text_file):
    
    # opening a new temp file
    temp_list = []
    
    eliminate = ['mspace', 'mtable', 'mathvariant', 'class', 'mpadded',
                'symmetric', 'fence', 'rspace', 'lspace', 'displaystyle', 'scriptlevel'
                'stretchy','form', 'movablelimits', 'maxsize', 'minsize', 'linethickness', 'mstyle']
    
    keep = ['mo', 'mi', 'mfrac', 'mn', 'mfrac','mrow']
    
    def count(eqn, e):
        c=0
        for word in eqn.split():
            if e in word:
                c+=1
        return c
    
    for t in mml_text_file:
        for eqn in t:
            for e in eliminate:
                if e in eqn:
                    c=count(eqn, e)
                    for _ in range(c):
                        idx = eqn.find(e)
                        # find the '<' just before the e
                        temp1 = eqn[:idx+1]
                        temp2 = eqn[idx+1:]
                        open_angle = [idx_open for idx_open, angle in enumerate(temp1) if angle == '<']
                        close_angle = [idx_close for idx_close, angle in enumerate(temp2) if angle == '>']
                        filtered = temp1[open_angle[-1]:]+temp2[:close_angle[0]+1]
                        flag = False
                        for k in keep:
                            if k in filtered:
                                  flag=True
                                  keep_token = k
                        if flag == True:
                            eqn = temp1[:open_angle[-1]]+f' <{keep_token}> '+temp2[close_angle[0]+1:]
                        else:
                            eqn = temp1[:open_angle[-1]]+temp2[close_angle[0]+1:]
    
            temp_list.append(eqn)

    return temp_list

def reduce_mml(temp_list):
    
    reduced_temp_list = []
    for eqn_ind,eqn in enumerate(temp_list):
        if '<mrow>' in eqn:
            f=''
            for F in eqn.split():
                f=f+F+' '
            idxs_open = []
            idxs_close = []
            for ind, i in enumerate(f.split()):
                if i == '<mrow>':
                    idxs_open.append(ind)s
                if i == '</mrow>':
                    idxs_close.append(ind)
            for o,c in zip(idxs_open, idxs_close):
                if len(f.split()[o:c+1])==3:
                    to_replace = ''
                    replace_with = ''
                    for fs in f.split()[o:c+1]:
                        to_replace+=fs+' '
                    replace_with = f.split()[o:c+1][1]+' '
                    f=f.replace(to_replace, replace_with)
            reduced_temp_list.append(f)
        
        else:
            f=''
            for F in eqn.split():
                f=f+F+' '
            reduced_temp_list.append(f)
    
    return reduced_temp_list
