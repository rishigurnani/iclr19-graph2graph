import sys
import argparse
import io
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--num_decode', type=int, default=20)
parser.add_argument('--sim_delta', type=float, default=0.4)
parser.add_argument('--prop_delta', type=float, default=0.9)
parser.add_argument('--total_n', type=int, default=0)
parser.add_argument('--mols_path', type=str)

args = parser.parse_args()

stdin = sys.stdin
num_decode = args.num_decode
sim_delta = args.sim_delta
prop_delta = args.prop_delta
total_n = args.total_n
mols_path = args.mols_path


import numpy as np
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

def diversity(pairs):
    diversity_values = []
    sources = set()
    decoded = {}
    # Build decoded dictionary that maps source polymers to the list of translated polymers
    for pair in pairs:
        source = pair[0]
        translated = pair[1]
        sources.add(source)
        if source in decoded:
            decoded[source].append(translated)
        else:
            decoded[source] = [translated]

    # Iterate over source molecules in dictionary and determine individual diversity scores
    for source in decoded:
        div = 0.0
        total = 0
        test_list = decoded[source]
        if len(test_list) > 1:
            for test in test_list:
                div += 1 - similarity(source, test)
                total += 1
            div /= total
        diversity_values.append(div)
    sources = list(sources)
    print 'Number of source polymers: ' + str(len(sources))
    return np.mean(diversity_values)

mols = []
for line in open(mols_path, 'r'):
    mols.append(line.strip())

data = []
start_append = False

len_d = total_n
for line in stdin:
    if 'Done' in line:
        start_append = True
    elif start_append:
            data.append(line.split())
#data = [line.split() for line in sys.stdin]
data = [(int(e),a,b,float(c),float(d)) for e,a,b,c,d in data]

n_mols = len_d

n_succ = 0.0

#load fp_df_fixed
fp_df = pd.read_csv('./fp_df_fixed.csv')
ignore_cols = [col for col in fp_df.keys() if col == 'ID' or 'Unnamed' in col]
fp_df = fp_df.drop(ignore_cols, axis=1)

def build_dict(data):
    '''
    Build a dictionary for all successful pairs
    '''
    d = {}
    for i in data:
        ind = i[0]
        x = i[1]
        y = i[2]
        sim = i[3]
        bg = i[4]
        if x in d:
            d[x].append((ind, y, sim, bg))
        else:
            d[x] = [(ind, y, sim, bg)]
    return d

data_d = build_dict(data)
new_targets = []
pairs = []
caught = []
for x, val in zip(data_d.keys(), data_d.values()):
    #print "Values: %s\n" %val
    good = [(ind,sim,bg,y) for ind,y,sim,bg in val if 1>sim>=sim_delta and bg>=prop_delta]
    #print "Good: %s\n" %good
    for tup in good:
        target = tup[3]
        ind = tup[0]
        #print "Target %s\n" %target
        if target not in mols:
            #print "target not in mols\n"
            fp = fp_df.iloc[ind, :].tolist()
            is_same = []
            for other in new_targets:
                #print "Other %s\n" %other[0:10]
                #print "New fp %s\n" %fp[0:10]
                result = (np.abs(np.subtract(other, fp)) < .001).all()
                #print "Bool %s\n" %result
                is_same.append(result)

            #print "Is same: %s" %is_same
            if not any(is_same):
                new_targets.append(fp)
                #print "new_targets %s\n" %[i[0:10] for i in new_targets]
                pairs.append((x, target))
                
                n_succ += 1
                print '%s %s %s' %(ind, x, target)
            else:
                #print "Target already in new targets\n"
                #print "new_targets %s\n" %[i[0:10] for i in new_targets]
                caught.append(target)

div_score = diversity(pairs)

print 'Evaluated on %d samples' % (n_mols)
print 'success rate', n_succ / (n_mols*num_decode)
print 'diversity score %s' %div_score