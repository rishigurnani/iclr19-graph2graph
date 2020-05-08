#mp = MULTI_PROPERTY
from __future__ import division
import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer="char")
import pickle as pkl
import sys
import argparse
import sys
import time
from joblib import Parallel, delayed

sys.path.append('/home/appls/machine_learning/PolymerGenome/src/common_lib')
sys.path.append('/home/rishi/py_scripts')
sys.path.append('/home/rgur/py_scripts')
import rishi_utils as ru

import auxFunctions as aF
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# Tanimoto similarity function
def similarity(a, b):
    if a is None or b is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

parser = argparse.ArgumentParser(description='specify test data size or percent')
parser.add_argument("--cutoff", type=float,
                    help="test data percent")
parser.add_argument("--load_sim", action='store_true', help="should similarities be loaded?")
parser.add_argument("--sim_path", type=str, help="filename where the similarity dict should be saved to or loaded from?")
parser.add_argument("--n_core", type=int, default=1, help="number of cores")
parser.add_argument("--data_path", type=str, help="path to raw data with target property values for each smiles")
parser.add_argument("--prop_cols", type=str,
                    help="list of property columns", nargs='+')
parser.add_argument("--source_thresh", type=float,
                    help="list of source thresholds", nargs='+')
parser.add_argument("--target_thresh", type=float,
                    help="list of target thresholds", nargs='+')
parser.add_argument("--strict", action='store_true', help="Should training pairs follow source_thresh strictly?")
parser.add_argument("--fuzzy_target", action='store_true', help="Should the target values be fuzzed?")

args = parser.parse_args()

n_prop = len(args.prop_cols)
print "Prop cols %s" %args.prop_cols
print "source_thresh %s" %args.source_thresh
print "target_thresh %s" %args.target_thresh

if len(args.source_thresh) != n_prop:
    raise ValueError('Not enough source thresholds specified')

if len(args.target_thresh) != n_prop:
    raise ValueError('Not enough target thresholds specified')

cutoff = args.cutoff

fp_all = ru.pd_load(args.data_path)
if 'id' not in fp_all.keys():
    n_rows = len(fp_all)
    ids = ['rd_%s' %i for i in range(n_rows)]
    fp_all['id'] = ids

source_operations = []
target_operations = []
for i in range(n_prop):
    if args.source_thresh[i] < args.target_thresh[i]:
        source_operations.append('<')
        target_operations.append('>')
    else:
        source_operations.append('>')
        target_operations.append('<')
        
def makeSourceTarget():
    include_cols = ['id', 'smiles'] + args.prop_cols
    if args.strict:
        source_cmd = ' & '.join(["(fp_all['%s'] %s %s)" %(args.prop_cols[i], source_operations[i], 
                                                          args.source_thresh[i]) for i in range(n_prop)])
    else:
        source_cmd = ' | '.join(["(fp_all['%s'] %s %s)" %(args.prop_cols[i], source_operations[i], 
                                                          args.source_thresh[i]) for i in range(n_prop)])   
    target_cmd = ' & '.join(["(fp_all['%s'] %s %s)" %(args.prop_cols[i], target_operations[i], 
                                                      args.target_thresh[i]) for i in range(n_prop)])
    exec( 'source = fp_all.loc[%s, %s]' %(source_cmd, include_cols) ) #make source
    exec( 'target = fp_all.loc[%s, %s]' %(target_cmd, include_cols) ) #make target
    return source, target

source, target = makeSourceTarget()
if args.fuzzy_target and len(target)/len(fp_all)<.0077:
    for ind,i in enumerate(target_operations):
            if abs(args.target_thresh[ind] - args.source_thresh[ind]) / args.target_thresh[ind] > .05:
                if i=='>':
                    args.target_thresh[ind] = args.target_thresh[ind]*.95
                elif i=='<':
                    args.target_thresh[ind] = args.target_thresh[ind]*1.05    
    source, target = makeSourceTarget()

def get_vec(p_id):
    '''
    Get feature vector from polymer_id
    '''
    row = fp_all[fp_all['id'] == p_id]
    return row.drop(['Unnamed: 0', 'id', 'smiles', 'dft_bandgap', 'norm_dft_bandgap', 'mean_dft_bandgap', 'std_dft_bandgap'], axis=1).to_numpy()

def multiply_one_smiles_old(s):
    '''
    Use naive implementation
    '''
    return (s*3).replace('[*]', '').replace('()', '')

def multiply_one_smiles_new(s):
    '''
    Use Chiho's implementation
    '''
    return aF.v2_multiply_smiles_star(s, number_of_multiplications = 3, debug=0)['extended_smiles']

def evaluate_pair(k):
    '''
    Evaluate similarity of pair and return
    '''
    s_smile, t_smile = source.loc[source['id'] == k[0], 'smiles'].item(), \
                          target.loc[target['id'] == k[1], 'smiles'].item()
    s_smile = multiply_one_smiles_new(s_smile)
    t_smile = multiply_one_smiles_new(t_smile)
    sim_val = similarity(s_smile, t_smile)
    return k, sim_val, s_smile, t_smile

# uncomment the following if needs to re-create pairs and store the similarities in the dictionary
if not args.load_sim:
    source_id, target_id = source['id'], target['id']
    pairs = list(product(source_id, target_id))
    sim_list = Parallel(n_jobs=args.n_core)(delayed(evaluate_pair)(k) for k in pairs)
    sim_dict = {val[0]:val[1:] for val in sim_list}
    del sim_list

    output_path = open(args.sim_path, 'wb')
    pkl.dump(sim_dict, output_path)
else:
    saved_path = open(args.sim_path, 'rb')
    sim_dict = pkl.load(saved_path)
sim_df = pd.DataFrame.from_dict(sim_dict, orient='index', columns=['similarity_score', 'source_smile', 'target_smile']).reset_index()
sim_df = sim_df.rename(columns={'index': 'source_target_id'})



# split according to the specified test_data_size or test_data_percent
def cal_cutoff(cut_off=None, test_data_percent=None, test_data_size=None):
    if cut_off:
        return cut_off
    if test_data_percent:
        return np.percentile(sim_df['similarity_score'], test_data_percent)
    if test_data_size:
        return np.percentile(sim_df['similarity_score'], test_data_size/len(sim_df)*100)

def fix_smiles(l):
    '''
    Returns list of valid SMILES without asterisks
    '''
    tmp = [i.replace('[*]', '') for i in l]
    return [i.replace('()', '') for i in tmp]

def split(cut_off):
    train = sim_df.loc[sim_df['similarity_score'] >= cut_off, :]
    source_smile_fix = train['source_smile'] #smiles is fixed upon creation of directory
    target_smile_fix = train['target_smile']

    source_in_train = train['source_smile'].unique().tolist()
    source_all = sim_df['source_smile'].unique().tolist()
    #source_in_train = fix_smiles(train['source_smile'].unique().tolist())
    #source_all = fix_smiles(sim_df['source_smile'].unique().tolist())

    test = [i for i in source_all if i not in source_in_train]
    outfile = open('trial_test_' + str(len(test)) + '.txt', "w")
    print >> outfile, "\n".join(str(i) for i in test)
    #print("\n".join(str(i) for i in test), file=outfile)
    outfile.close()

    #save train
    train_list = [i + ' ' + j for i,j in zip(train['source_smile'].tolist(), train['target_smile'].tolist())]
    outfile = open('trial_train_' + str(len(train)) + '.txt', "w")
    print >> outfile, "\n".join(str(i) for i in train_list)
    #print("\n".join(str(i) for i in train_list), file=outfile)
    outfile.close()

    #save mols
    mols = [multiply_one_smiles_new(s) for s in fp_all['smiles'].tolist()]
    #mols = fix_smiles([i + i+ i for i in fp_all['smiles'].tolist()])
    #mols = list(set(source_smile_fix).union(set(target_smile_fix)))
    outfile = open("mols.txt", "w")
    print >> outfile, "\n".join(str(i) for i in mols)
    #print("\n".join(str(i) for i in mols), file=outfile)
    outfile.close()

    return train, test

def main():
    split(cutoff)


if __name__ == '__main__':
    start = time.time()
#     parser = argparse.ArgumentParser(description='specify test data size or percent')
#     parser.add_argument("--cutoff", type=float,
#                         help="test data percent")
#     parser.add_argument("--percent", type=int,
#                         help="test data percent")
#     parser.add_argument("--size", type=int,
#                         help="test data size")
#     parser.add_argument("--load_sim", type=bool, help="should similarities be loaded?")
#     parse.add_argument("--sim_path", type=str, help="filename where the similarity dict should be saved to or loaded from?=")

    main()
    end = time.time()
    print "Time Elapsed: %s" %(end-start)