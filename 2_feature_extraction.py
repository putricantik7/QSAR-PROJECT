import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from tqdm import tqdm
import os

def extract_features():
    input_file = 'data_clean.csv'
    if not os.path.exists(input_file):
        print(f"File {input_file} tidak ditemukan. Harap jalankan 1_data_collection.py terlebih dahulu.")
        return

    print("[1/3] Memuat data...")
    df = pd.read_csv(input_file)
    print(f"      Total senyawa: {len(df)}")

    # Parameter Morgan Fingerprint
    radius = 2
    nBits = 1024

    X_morgan = []
    X_2d = []
    valid_indices = []

    print("[2/3] Mengekstraksi Fitur (Morgan Fingerprints & 2D Descriptors)...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        smiles = row['canonical_smiles']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # 1. Morgan Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fp_arr = np.zeros((1,))
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, fp_arr)
            
            # 2. 2D Descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            
            X_morgan.append(fp_arr)
            X_2d.append([mw, logp, hbd, hba, tpsa])
            valid_indices.append(index)
        else:
            print(f"      Gagal membaca SMILES pada indeks {index}")

    # Mengonversi list ke DataFrame
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    
    # DataFrame Fingerprint
    col_morgan = [f'Morgan_{i}' for i in range(nBits)]
    df_morgan = pd.DataFrame(X_morgan, columns=col_morgan)
    
    # DataFrame 2D
    col_2d = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    df_2d = pd.DataFrame(X_2d, columns=col_2d)
    
    # Menggabungkan data asli (hanya ID dan p_activity) dengan fitur
    df_final = pd.concat([df_valid[['molecule_chembl_id', 'canonical_smiles', 'p_activity']], df_morgan, df_2d], axis=1)

    print("[3/3] Menyimpan dataset fitur...")
    output_file = 'dataset_features.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Selesai! Fitur berhasil disimpan ke {output_file} dengan dimensi {df_final.shape}")

if __name__ == "__main__":
    extract_features()
