import pandas as pd
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='specify test data size or percent')
parser.add_argument("--cutoff", type=float,
                    help="test data percent")
parser.add_argument("--percent", type=int,
                    help="test data percent")
parser.add_argument("--size", type=int,
                    help="test data size")

parser.add_argument("--sim_path", type=str,
                    help="path to similarity file")

parser.add_argument("--fp_path", type=str,
                    help="path to fingerprint file")

sim_df = pd.read_csv('output_sim.csv', index_col=False).reset_index()
sim_df = sim_df.rename(columns={'index': 'source_target_id'})
fp_all = pd.read_csv('fp_all_data.csv')

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
    source_smile_fix = fix_smiles(train['source_smile'])
    train['source_smile'] = source_smile_fix

    target_smile_fix = fix_smiles(train['target_smile'])
    train['target_smile'] = target_smile_fix

    source_in_train = fix_smiles(train['source_smile'].unique().tolist())

    source_all = fix_smiles(sim_df['source_smile'].unique().tolist())

    test = [i for i in source_all if i not in source_in_train]
    #save test
    outfile = open('trial_test_' + str(len(test)) + '.txt', "w")
    # print >> outfile, "\n".join(str(i) for i in test)
    print("\n".join(str(i) for i in test), file=outfile)
    outfile.close()

    #save train
    train_list = [i + ' ' + j for i,j in zip(train['source_smile'].tolist(), train['target_smile'].tolist())]
    outfile = open('trial_train_' + str(len(train)) + '.txt', "w")
    # print >> outfile, "\n".join(str(i) for i in train_list)
    print("\n".join(str(i) for i in train_list), file=outfile)
    outfile.close()

    #save mols
    mols = fix_smiles([i + i+ i for i in fp_all['smiles'].tolist()])
    outfile = open("mols.txt", "w")
    # print >> outfile, "\n".join(str(i) for i in mols)
    print("\n".join(str(i) for i in mols), file=outfile)
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
    main()





