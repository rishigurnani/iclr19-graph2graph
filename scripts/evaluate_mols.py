#return number of invalid molecules
from rdkit import Chem
import sys
import argparse

parser = argparse.ArgumentParser(description='specify test data size or percent')
parser.add_argument("--mols", type=str,
                    help="path to mols file")


args = parser.parse_args()

# Write functions

def smilesList(path):
    '''
    Generate list of SMILES from path
    '''
    with open(path) as f:
        data = [line.strip() for line in f]
    return data

def getNAtoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        Chem.AddHs(mol)
        return None
    except:
        print smiles
        return smiles

# Test functions

l = smilesList(args.mols)
out = [getNAtoms(s) for s in l]
bad_smiles = [s for s in out if s != None]

print "%s invalid monomers" %len(bad_smiles)