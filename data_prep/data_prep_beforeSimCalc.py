import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import json


fp_all = pd.read_csv('fp_all_data.csv')
ids = fp_all['id']

def get_fp_jaccard(smiles): # smiles already fixed
    num_bits = 2048
    if smiles is None:
        return np.zeros(num_bits)
    amol = Chem.MolFromSmiles(smiles)
    if amol is None:
        return np.zeros(num_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=num_bits, useChirality=False)
    fp = np.array(fp).astype(int)
    return fp


def get_vec_cosine(p_id):
    '''
    Get feature vector from polymer_id
    '''
    row = fp_all[fp_all['id'] == p_id]
    vec = row.drop(['Unnamed: 0', 'id', 'smiles', 'dft_bandgap', 'norm_dft_bandgap', 'mean_dft_bandgap', 'std_dft_bandgap'], axis=1).to_numpy()
    return vec[0].astype(np.double)

fp_info = {}
vec_info = {}

for id in ids:
    bandgap = fp_all.loc[fp_all['id'] == id, 'dft_bandgap'].item()
    smiles = fp_all.loc[fp_all['id'] == id, 'smiles'].item()
    smiles = smiles.replace('[*]', '') 
    print('current id is {}'.format(id))
    
    print('------get_fp_jaccard--------')
    fp = get_fp_jaccard(smiles)
    print(fp, np.count_nonzero(fp), fp.shape)
    fp_info[id] = [fp.tolist(), bandgap, smiles*3]

    print('------get_vec_cosine--------')
    vec = get_vec_cosine(id)
    # print(vec)
    print(vec.shape)
    vec_info[id] = [vec.tolist(), bandgap, smiles*3]

fp_info = [{'p_id': k, 'fp': v[0], 'dft_bandgap': v[1], 'smiles': v[2]} for k, v in fp_info.items()]
with open('fp_info.json', 'w') as f:
    json.dump(fp_info, f)


vec_info = [{'p_id': k, 'fp': v[0], 'dft_bandgap': v[1], 'smiles': v[2]} for k, v in vec_info.items()]
with open('vec_info.json', 'w') as f:
    json.dump(vec_info, f)

