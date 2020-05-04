import pandas as pd
import os
import sys
#from props import *
from poly_diversity import similarity
import argparse
import time

parser = argparse.ArgumentParser(description='specify test data size or percent')

#some functions which satisfy the following properties:
# 1) read from fp_df.csv
# 2) return list of property predictions
# 3) have function named 'main' which returns necessary data
parser.add_argument("--predictors", type=str,
                    help="list of python files to run for property evaluation", nargs='+') 
# parser.add_argument("--prop_names", type=str,
#                     help="list of names for each property", nargs='+')
args = parser.parse_args()

n_prop = len(args.predictors)
path_to_append = []
modules = []
for path in args.predictors:
    spl = path.split('/')
    sys.path.append('/'.join(spl[0:-1]))
    modules.append(spl[-1].split('.py')[0]) #exclude .py from module name

functions = []
for i, module in enumerate(modules):
    functions.append( 'main_%s' %i )
    exec( 'from %s import main as main_%s' %(module, i) )

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
    save_smiles_df(l)

    create_fp_input()

    run_fp()

    ind2ID_map = fix_fp()

    create_pred_input(model_path)

    run_pred()

    return ind2ID_map, get_pred()

#perform tasks
ys = []
xs = []
lines = []
for line in sys.stdin:
    if line not in lines:
        lines.append(line)
        x,y = line.split()
        ys.append(y)
        xs.append(x)
        if y == "None": y = None


            
save_smiles_df(ys)

create_fp_input()

run_fp()

ind2ID_map = fix_fp()
#ind2ID_map, outs = get_all_preds(ys)

props = []
for f in functions:
    exec('out = %s()' %f)
    props.append( out )

zipped_props = zip(*props)
del props

def tup_to_str(tup):
    l = list(tup)

    l = [str(i) for i in l]

    return ' '.join(l)

for ind, prop in enumerate(zipped_props):
    ID = ind2ID_map[ind]
    #x = xs[ind]
    #y = ys[ind]
    x = xs[ID]
    y = ys[ID]
    sim2D = similarity(x, y)
    prop_string = tup_to_str(prop)
    try:
        print ind, x, y, sim2D, prop_string
    except:
        print ind, x, y, sim2D, 0.0