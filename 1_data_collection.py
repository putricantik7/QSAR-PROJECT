import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors

def collect_data():
    print("[1/5] Mencari target Staphylococcus aureus di ChEMBL...", flush=True)
    target = new_client.target
    target_query = target.search('Staphylococcus aureus')
    targets = pd.DataFrame.from_dict(target_query)
    
    sa_target = targets[targets['target_type'] == 'ORGANISM']
    if sa_target.empty:
        sa_target = targets.iloc[0]
    else:
        sa_target = sa_target.iloc[0]
        
    target_chembl_id = sa_target['target_chembl_id']
    print(f"      Target terpilih: {sa_target['pref_name']} (ID: {target_chembl_id})", flush=True)

    print("[2/5] Mengambil 2500 data bioaktivitas S. aureus (MIC/IC50)...", flush=True)
    activity = new_client.activity
    # Gunakan slicing untuk membatasi query langsung dari server (hindari 500 error)
    query = activity.filter(target_chembl_id=target_chembl_id).filter(standard_type__in=['MIC', 'IC50'])[0:3000]
    
    collected_data = []
    seen_smiles = set()
    
    # Batasi hingga 2500 senyawa unik untuk efisiensi komputasi dan menghindari timeout
    TARGET_COUNT = 2500
    
    for item in query:
        smiles = item.get('canonical_smiles')
        val = item.get('standard_value')
        
        if not smiles or val is None or val == '':
            continue
            
        if smiles in seen_smiles:
            continue
            
        # Pastikan valid structure
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            collected_data.append(item)
            seen_smiles.add(smiles)
            
        if len(collected_data) % 500 == 0:
            print(f"      [Progress] Mengumpulkan {len(collected_data)} / {TARGET_COUNT} senyawa...", flush=True)
            
        if len(collected_data) >= TARGET_COUNT:
            break

    print(f"      Berhasil mengumpulkan {len(collected_data)} senyawa.", flush=True)
    df = pd.DataFrame(collected_data)

    print("[3/5] Pra-pemrosesan data...", flush=True)
    df['standard_value'] = df['standard_value'].astype(float)
    
    print("[4/5] Mengonversi nilai aktivitas ke pIC50 / pMIC...", flush=True)
    def convert_to_p_value(row):
        value = float(row['standard_value'])
        unit = row['standard_units']
        
        if unit == 'nM':
            value_m = value * 1e-9
        elif unit in ['ug.mL-1', 'ug/mL']:
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            mw = Descriptors.MolWt(mol)
            value_m = (value * 1e-3) / mw
        elif unit == 'uM':
            value_m = value * 1e-6
        else:
            value_m = value * 1e-9
            
        if value_m > 0:
            return -np.log10(value_m)
        else:
            return np.nan

    df['p_activity'] = df.apply(convert_to_p_value, axis=1)
    df = df.dropna(subset=['p_activity'])

    df_final = df[['molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_value', 'standard_units', 'p_activity']]
    
    print("[5/5] Menyimpan data...", flush=True)
    output_file = 'data_clean.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Selesai! Data berhasil disimpan ke {output_file} (Jumlah senyawa: {len(df_final)})", flush=True)

if __name__ == "__main__":
    collect_data()
