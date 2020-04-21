import pandas as pd
import os
import sys
from props import *

import time

args = sys.argv

model_path = args[1] #path to bandgap model

def polymerize(s):
    '''
    Turn monomer SMILES into polymer SMILES
    '''
    return '[*]' + s + '[*]'

def save_smiles_df(l):
    '''
    Save df from list of SMILES
    '''
    n_smiles = len(l)
    ID = list(range(n_smiles))
    pd.DataFrame({"ID": ID, "SMILES": l}).to_csv('./smiles_df.csv')

def create_fp_input():
#     print('file_dataset = ./smiles_df.csv')
#     print('col_smiles = SMILES')
#     print('col_X = aT bT m e')
#     print('col_id = ID')
#     print('file_fingerprint = fp_df.csv')
#     print('polymer_fp_version = 2')
#     print('ismolecule = 0')
#     print('drop_failed_rows = 0')
#     print('ncore = 18')
    f = ('file_dataset = ./smiles_df.csv\n'
    'col_smiles = SMILES\n'
    'col_X = aT bT m e\n'
    'col_id = ID\n'
    'file_fingerprint = fp_df.csv\n'
    'polymer_fp_version = 2\n'
    'ismolecule = 0\n'
    'drop_failed_rows = 0\n'
    'ncore = 18\n')
    text_file = open('fp_input', "w")
    text_file.write(f)
    text_file.close()

def run_fp():
    os.system('fp fp_input')

def fix_fp():
    df = pd.read_csv('fp_df.csv')
    #df = df.iloc[df.dropna().index].reset_index().drop('index', axis=1)
<<<<<<< HEAD
    df = df.iloc[df.dropna().index]
    df = df.set_index('ID')
    #use_cols = [col for col in df.keys() if col != 'ID' and 'Unnamed' not in col]
    #df.drop_duplicates(subset=use_cols, inplace=True)
    df.to_csv('fp_df_fixed.csv')
=======
    bad_inds = df.dropna().index
    df = df.iloc[bad_inds]
    #df = df.set_index('ID')
    df = df.reset_index().drop('index', axis=1)
    ind2ID_map = {k:v for k,v in zip(df.index.tolist(), df['ID'].tolist())}
    #use_cols = [col for col in df.keys() if col != 'ID' and 'Unnamed' not in col]
    #df.drop_duplicates(subset=use_cols, inplace=True)
    df = df.round(decimals=5)
    df.to_csv('fp_df_fixed.csv')
    return ind2ID_map
>>>>>>> dc0a39c35bb6a3dd7ac5c6d2aa18f3a1dabebed0
    
def create_pred_input(model_path):
    f = ('file_model = ' + model_path + '\n'
    'file_fingerprint  = fp_df_fixed.csv\n'
    'file_output = output.csv\n'
    )
    text_file = open('pred_input', "w")
    text_file.write(f)
    text_file.close()    

def run_pred():
    try:
        os.system('predict pred_input > log')
    except:
        pass

def get_pred():
    
    df = pd.read_csv('output.csv')
    normed = df['y'].tolist()
    original_scale = [(val*1.67668) + 4.34115 for val in normed]
    return zip(df.index.to_list(), original_scale)

def get_all_preds(l):
    poly_l = [polymerize(i) for i in l]
    save_smiles_df(poly_l)

    create_fp_input()

    run_fp()

<<<<<<< HEAD
    fix_fp()
=======
    ind2ID_map = fix_fp()
>>>>>>> dc0a39c35bb6a3dd7ac5c6d2aa18f3a1dabebed0
    
    create_pred_input(model_path)

    run_pred()

<<<<<<< HEAD
    return get_pred()    
=======
    return ind2ID_map, get_pred()    
>>>>>>> dc0a39c35bb6a3dd7ac5c6d2aa18f3a1dabebed0

#perform tasks
ys = []
xs = []
<<<<<<< HEAD
sim2Ds = []
=======
>>>>>>> dc0a39c35bb6a3dd7ac5c6d2aa18f3a1dabebed0
for line in sys.stdin:
    x,y = line.split()
    ys.append(y)
    xs.append(x)
    if y == "None": y = None
<<<<<<< HEAD
    #sim2Ds.append(similarity(x, y))

outs = get_all_preds(ys)
=======

ind2ID_map, outs = get_all_preds(ys)
>>>>>>> dc0a39c35bb6a3dd7ac5c6d2aa18f3a1dabebed0

for out in outs:
    ind = out[0]
    bg = out[1]
<<<<<<< HEAD
    x = xs[ind]
    y = ys[ind]
=======
    ID = ind2ID_map[ind]
    #x = xs[ind]
    #y = ys[ind]
    x = xs[ID]
    y = ys[ID]
>>>>>>> dc0a39c35bb6a3dd7ac5c6d2aa18f3a1dabebed0
    sim2D = similarity(x, y)
    try:
        print ind, x, y, sim2D, bg
    except:
        print ind, x, y, sim2D, 0.0