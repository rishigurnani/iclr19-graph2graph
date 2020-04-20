import pandas as pd
import numpy as np
# from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(analyzer="char")
import pickle as pkl
from itertools import product
from sklearn.metrics import jaccard_score
# import sys
# import argparse
# from props import *

fp_all = pd.read_csv('fp_all_data.csv')
source = fp_all.loc[fp_all['dft_bandgap'] < 4, ['id', 'smiles', 'dft_bandgap']]
target = fp_all.loc[fp_all['dft_bandgap'] > 6, ['id', 'smiles', 'dft_bandgap']]

# num_source = len(source)  # 246
# num_target = len(target)  # 1027


import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def similarity(a, b): # Tanimoto similarity
    if a is None or b is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)

    print('-----------test1: confirm jaccard_score is performing same way as the DataStructs.TanimotoSimilarity-----------')
    a, b = np.array(fp1), np.array(fp2)
    j = jaccard_score(a, b)
    print('jaccard score is {}'.format(j))
    t = DataStructs.TanimotoSimilarity(fp1, fp2)
    print('TanimotoSimilarity is {}'.format(t))
    assert abs(j-t) <= 1e-10, 'j and t should be similar, j is {}, t is {}'.format(j, t)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_vec(p_id):
    '''
    Get feature vector from polymer_id
    '''
    row = fp_all[fp_all['id'] == p_id]
    return row.drop(['Unnamed: 0', 'id', 'smiles', 'dft_bandgap', 'norm_dft_bandgap', 'mean_dft_bandgap', 'std_dft_bandgap'], axis=1).to_numpy()

# uncomment the following if needs to re-create pairs and store the similarities in the dictionary
source_id, target_id = source['id'], target['id']
pairs = list(product(source_id, target_id))
sim_dict = {}

test_count = 0
for k in pairs:
   s_smile, t_smile = source.loc[source['id'] == k[0], 'smiles'].item(), \
                      target.loc[target['id'] == k[1], 'smiles'].item()
#    s_vec, t_vec = get_vec(k[0]), get_vec(k[1]) # notice the shape, it's 1*1
#    sim = cosine_similarity(s_vec, t_vec)[0][0]

   s_smile = s_smile.replace('[*]', '')
   t_smile = t_smile.replace('[*]', '')
   sim_val = similarity(s_smile, t_smile)

   test_count += 1
   if test_count == 1:
       break

print('-----------test2: confirm scala code is performing same way as the scikit-learn cosine_similarity-----------')
s = np.array([[0.5, 0.6, 0]])
t = np.array([[0.7, 0, 0.8]])
print('test cosine similarity is {}'.format(cosine_similarity(s, t)[0][0]))

print('-----------test3: confirm scala code is performing same way as the scikit-learn jaccard_score-----------')
s = '1 0 0 1 1 1 0 1 1 0 1 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 0 0'
t = '0 1 0 0 1 1 1 1 0 0 1 0 1 1 0 1 1 0 0 1 0 0 1 1 0 0 1 0 1 1 0 1'
s = '0 0 0 0 0'
t = '0 0 0 1 0'
s = np.array([int(i) for i in s.split(' ')])
t = np.array([int(i) for i in t.split(' ')])
print(list(s))
print(list(t))
print('jaccard score is {}'.format(jaccard_score(s, t)))
