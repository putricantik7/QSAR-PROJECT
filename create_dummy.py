import pandas as pd
import numpy as np
import random

# Daftar SMILES bervariasi (termasuk beberapa kerangka flavonol dan senyawa antibakteri umum)
smiles_list = [
    # Flavonols and flavonoids
    "O=C1C(O)=C(c2ccccc2)Oc3ccccc13", # 3-hydroxyflavone (flavonol core)
    "O=C1C(O)=C(c2ccc(O)cc2)Oc3cc(O)cc(O)c13", # Kaempferol
    "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13", # Quercetin
    "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)ccc13", # Fisetin
    "O=C1C(O)=C(c2ccc(O)c(OC)c2)Oc3cc(O)cc(O)c13", # Isorhamnetin
    "O=C1CC(c2ccc(O)cc2)Oc3cc(O)cc(O)c13", # Naringenin
    "O=C1C=C(c2ccc(O)cc2)Oc3cc(O)cc(O)c13", # Apigenin
    "O=C1C=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13", # Luteolin
    
    # Generic structures to pad the dataset
    "Cc1ccccc1", "c1ccccc1", "OC(=O)c1ccccc1", "Nc1ccccc1", 
    "CC(=O)Oc1ccccc1C(=O)O", "CC(C)c1ccc(C)cc1", "Oc1ccccc1",
    "Cc1ccc(O)cc1", "Clc1ccccc1", "Brc1ccccc1", "Fc1ccccc1",
    "COc1ccccc1", "CCOc1ccccc1", "O=Cc1ccccc1", "CC(=O)c1ccccc1",
    "CCC(C)(C)C", "CCCC", "CCCCCC", "c1ccncc1", "c1ccsc1",
    "c1ccoc1", "n1ccccc1", "c1cncnc1", "c1c(O)cc(O)cc1O",
    "CC1CCC2C(C1)CCC3C2CCC4(C3CCC4(C)C)C", # Steroid core
    "C1CCCCC1", "C1CCCC1", "C1CCC1", "CC(C)(C)C",
]

# Generate random padding to reach 200 compounds (modifying base SMILES slightly)
base_smiles = smiles_list.copy()
for i in range(170):
    base = random.choice(base_smiles)
    # just append a methyl group somewhere or append to a ring naively? 
    # to avoid invalid smiles, we just reuse them but add dummy activity.
    # Actually, RDKit will drop duplicates. Let's just create a bunch of linear/branched alkanes and simple aromatics.
    chain = "C" * random.randint(1, 10)
    if random.random() > 0.5:
        chain += "O"
    if random.random() > 0.5:
        chain += "N"
    smiles_list.append(chain)

data = []
for i, smiles in enumerate(smiles_list):
    # Random activity between 3.0 and 9.0 (pIC50 scale)
    # We add a slight correlation: length of string correlates with activity loosely
    p_act = 4.0 + (len(smiles) % 5) + random.uniform(-0.5, 0.5)
    
    data.append({
        'molecule_chembl_id': f'DUMMY_CHEMBL_{i}',
        'canonical_smiles': smiles,
        'standard_type': 'MIC',
        'standard_value': 10 ** (-p_act) * 1e9, # nM
        'standard_units': 'nM',
        'p_activity': p_act
    })

df = pd.DataFrame(data)

# Filter out invalid SMILES just in case
from rdkit import Chem
df['valid'] = df['canonical_smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
df = df[df['valid'] == True].drop(columns=['valid'])

df.to_csv('data_clean.csv', index=False)
print(f"Dummy dataset created with {len(df)} valid compounds.")
