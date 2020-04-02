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

def diversity(pairs, num_decode):
    diversity_values = []
    num_test = set()
    for i in xrange(0, len(pairs), num_decode):
        data = pairs[i:i + num_decode]
        div = 0.0
        total = 0
        for i in xrange(len(data)):
            num_test.add(data[i][0])
            for j in xrange(i + 1, len(data)):
                div += 1 - similarity(data[i][1], data[j][1])
                total += 1
        div /= total
        diversity_values.append(div)
    num_test = list(num_test)
    print 'Number of test polymers: ' + str(len(num_test))
    return np.mean(diversity_values)

if __name__ == "__main__":
    pairs = [('COc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)CCOc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)CCOc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)C', 'Cc1csc(CNC(=O)c2ccc(Cl)c(N3CCCC3=O)c2)n1Cc1csc(CNC(=O)c2ccc(Cl)c(N3CCCC3=O)c2)n1Cc1csc(CNC(=O)c2ccc(Cl)c(N3CCCC3=O)c2)n1'),
    ('COc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)CCOc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)CCOc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)C', 'Cc1cc(NC2CCCC2)nc([C@@H]2CCCN2S(C)(=O)=O)n1Cc1cc(NC2CCCC2)nc([C@@H]2CCCN2S(C)(=O)=O)n1Cc1cc(NC2CCCC2)nc([C@@H]2CCCN2S(C)(=O)=O)n1'),
    ('COc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)CCOc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)CCOc1ccc(Cl)cc1C[C@@]1([NH3+])CCCCC1(C)C', 'CC1(C)CCCC[C@H]1NC(=O)c1scnc1C1CC1CC1(C)CCCC[C@H]1NC(=O)c1scnc1C1CC1CC1(C)CCCC[C@H]1NC(=O)c1scnc1C1CC1')]
    num_decode = 3
    print diversity(pairs ,num_decode)