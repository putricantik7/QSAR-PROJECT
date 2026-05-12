import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

# [MODIFIKASI] Ubah default model_file dari 'mlr_model.pkl' menjadi 'rf_model.pkl'
def predict_candidate(smiles, model_file='rf_model.pkl'):
    print(f"Memuat model dari {model_file}...")
    try:
        saved_data = joblib.load(model_file)
        model = saved_data['model']
        selector = saved_data['selector']
    except FileNotFoundError:
        print(f"Error: {model_file} tidak ditemukan. Latih model terlebih dahulu.")
        return
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("SMILES tidak valid!")
        return
        
    print(f"Molekul valid. Mengekstraksi fitur...")
    # Morgan Fingerprints
    radius = 2
    nBits = 1024
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    fp_arr = np.zeros((1,))
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, fp_arr)
    
    # 2D Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Gabungkan fitur
    features_2d = np.array([mw, logp, hbd, hba, tpsa])
    features = np.concatenate([fp_arr, features_2d]).reshape(1, -1)
    
    # Transformasi VarianceThreshold
    features_filtered = selector.transform(features)
    
    # Prediksi
    prediction = model.predict(features_filtered)[0]
    
    print("\n--- HASIL PREDIKSI ---")
    print(f"SMILES Kandidat: {smiles}")
    print(f"Prediksi pActivity (pMIC/pIC50): {prediction:.4f}")
    
    # Konversi balik (kasar) ke MIC (nM) jika diinginkan: MIC_nM = 10^(-pActivity) * 1e9
    # (Tergantung standarisasi saat data collection)
    print("----------------------\n")

if __name__ == "__main__":
    # Contoh SMILES Quercetin (salah satu flavonol)
    quercetin_smiles = "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13"
    print("Menguji kandidat Quercetin...")
    predict_candidate(quercetin_smiles)