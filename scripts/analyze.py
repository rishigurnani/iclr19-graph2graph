import sys
import argparse
import io
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--num_decode', type=int, default=20)
parser.add_argument('--sim_delta', type=float, default=0.4)
parser.add_argument("--prop_targets", type=str, help="list of property targets", nargs='+', required=True)
#parser.add_argument("--prop_targets", type=list, help="list of property targets")
parser.add_argument('--total_n', type=int, default=0)
parser.add_argument('--mols_path', type=str)
args = parser.parse_args()

n_prop = len(args.prop_targets)
stdin = sys.stdin
num_decode = args.num_decode
sim_delta = args.sim_delta
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

#data = [(int(e),a,b,float(c),float(d)) for e,a,b,c,d in data] #ind, source, target, sim, props

float_str = ','.join(['float(prop%s)' %i for i in range(n_prop)])
parse_str = ','.join(['prop%s' %i for i in range(n_prop)])
# if len(data[0]) < 4+n_prop:
#     raise ValueError('Too few arguments in data')
cmd = 'data = [( int(ind),x,y,float(sim),%s ) for ind,x,y,sim,%s in data]' %( float_str,parse_str )  #ind, source, target, sim, props
exec(cmd)

n_mols = len_d

n_succ = 0.0

#load fp_df_fixed
try:
    fp_df = pd.read_csv('./fp_df_fixed.csv')
    ignore_cols = [col for col in fp_df.keys() if col == 'ID' or 'Unnamed' in col]
    fp_df = fp_df.drop(ignore_cols, axis=1)

    def build_dict(data):
        '''
        Build a dictionary for all successful pairs
        '''
        d = {}
        for i in data:
            x_ind = 1
            x = i[x_ind]
            l = list(i)
            small_l = [val for ind,val in enumerate(l) if ind != x_ind] #remove x from list
            val = tuple(small_l)
            if x in d:
                d[x].append(val)
            else:
                d[x] = [val]
        return d

    data_d = build_dict(data)
    new_targets = []
    pairs = []
    caught = []
    for x, val in zip(data_d.keys(), data_d.values()):
        #print "Values: %s\n" %val
        conditions = ' and '.join(['prop%s%s'%( i,args.prop_targets[i] ) for i in range(n_prop)])
        cmd2 = 'good = [(ind,sim,%s,y) for ind,y,sim,%s in val if sim>=sim_delta and %s]'%( parse_str,parse_str, conditions)
        exec(cmd2)
        #print "Good: %s\n" %good
        for tup in good:
            target = tup[-1]
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
except:
    print 'Evaluated on %d samples' % (n_mols)
    print 'success rate 0.0'
    print 'diversity score nan' 