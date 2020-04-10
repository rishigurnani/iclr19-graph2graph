import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer="char")
import pickle as pkl
import sys
import argparse
from props import *

fp_all = pd.read_csv('./data/fp_all_data.csv')
source = fp_all.loc[fp_all['dft_bandgap'] < 2.5, ['id', 'smiles', 'dft_bandgap']]
target = fp_all.loc[fp_all['dft_bandgap'] > 4, ['id', 'smiles', 'dft_bandgap']]

# num_source = len(source)  # 246
# num_target = len(target)  # 1027

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
for k in pairs:
   s_smile, t_smile = source.loc[source['id'] == k[0], 'smiles'].item(), \
                      target.loc[target['id'] == k[1], 'smiles'].item()
   #s_vec, t_vec = get_vec(k[0]), get_vec(k[1])
   #sim = cosine_similarity(s_vec, t_vec)[0][0]
   s_smile = s_smile.replace('[*]', '')
   t_smile = t_smile.replace('[*]', '')
   sim_val = similarity(s_smile, t_smile)
   sim_dict[k] = [sim_val, s_smile*3, t_smile*3]
output_path = open('similarities.pkl', 'wb')
pkl.dump(sim_dict, output_path)

saved_path = open('similarities.pkl', 'rb')
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
    #test = sim_df.loc[sim_df['similarity_score'] < cut_off, :]
    train = sim_df.loc[sim_df['similarity_score'] >= cut_off, :]
    source_smile_fix = fix_smiles(train['source_smile'])
    train['source_smile'] = source_smile_fix

    target_smile_fix = fix_smiles(train['target_smile'])
    train['target_smile'] = target_smile_fix

    source_in_train = fix_smiles(train['source_smile'].unique().tolist())

    source_all = fix_smiles(sim_df['source_smile'].unique().tolist())

    test = [i for i in source_all if i not in source_in_train]
    #train.to_csv('trial_train_' + str(len(train)) + '.csv', index=False)
    #test.to_csv('trial_test_' + str(len(test)) + '.csv', index=False)
#     print('result: cut_off {}, size of test data {}, percent of test data {}'.format(
#         cut_off, len(test), len(test)/len(sim_df)))

    #save test
    #print("Hello stackoverflow!", file=open("output.txt", "a"))
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
    mols = fix_smiles([i + i+ i for i in fp_all['smiles'].tolist()])
    outfile = open("mols.txt", "w")
    print >> outfile, "\n".join(str(i) for i in mols)
    #print("\n".join(str(i) for i in mols), file=outfile)
    outfile.close()

    return train, test

def main():
    args = parser.parse_args()
    cutoff = args.cutoff
    percent = args.percent
    size = args.size
    if not cutoff and not percent and not size:
        print('no cutoff or test_data_percent or test_data_size specified, exiting')
        sys.exit()
    cut_off = cal_cutoff(cutoff, percent, size)
    split(cut_off)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='specify test data size or percent')
    parser.add_argument("--cutoff", type=float,
                        help="test data percent")
    parser.add_argument("--percent", type=int,
                        help="test data percent")
    parser.add_argument("--size", type=int,
                        help="test data size")
    main()





