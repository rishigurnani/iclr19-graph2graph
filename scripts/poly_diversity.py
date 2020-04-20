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
