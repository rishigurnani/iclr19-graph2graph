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
            decoded[curr_source] = [translated]
    print(decoded)
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
    print 'Number of test polymers: ' + str(len(sources))
    return np.mean(diversity_values)

if __name__ == "__main__":
    pairs = [('CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', 'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1'),
    ('CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', 'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1OC[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1OC[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O'),
    ('CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', 'O=C(NCc1cc(F)cc(F)c1)c1cccnc1O=C(NCc1cc(F)cc(F)c1)c1cccnc1O=C(NCc1cc(F)cc(F)c1)c1cccnc1'),
    ('COc1cc(Cl)ccc1C(=O)OC[C@H]1CCCCO1', 'C[C@H](Oc1cc(Cl)ccc1Cl)C(=O)NC[C@@H]1CCCO1')]
    print diversity(pairs)