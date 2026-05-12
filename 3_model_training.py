import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # [MODIFIKASI] Menggunakan Random Forest
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    input_file = 'dataset_features.csv'
    if not os.path.exists(input_file):
        print(f"File {input_file} tidak ditemukan. Harap jalankan 2_feature_extraction.py terlebih dahulu.")
        return

    print("[1/5] Memuat dataset fitur...")
    df = pd.read_csv(input_file)
    
    # Fitur (X) dimulai setelah kolom p_activity
    feature_start_idx = df.columns.get_loc('p_activity') + 1
    
    X = df.iloc[:, feature_start_idx:].values
    y = df['p_activity'].values
    
    print(f"      Dimensi X (Fitur): {X.shape}")
    print(f"      Dimensi y (Target): {y.shape}")

    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01) 
    X_filtered = selector.fit_transform(X)
    print(f"      Dimensi X setelah VarianceThreshold: {X_filtered.shape}")

    print("[2/5] Membagi dataset (Train & Test 80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
    
    print("[3/5] Melatih model Random Forest Regressor...")
    # [MODIFIKASI] Inisiasi model Random Forest dengan 100 pohon (trees)
    model = RandomForestRegressor(n_estimators=100, random_state=42) 
    model.fit(X_train, y_train)

    print("[4/5] Evaluasi Model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    print(f"      [Train] R2: {r2_train:.4f} | MSE: {mse_train:.4f}")
    print(f"      [Test]  R2: {r2_test:.4f} | MSE: {mse_test:.4f}")

    print("[5/5] Visualisasi dan Penyimpanan Model...")
    plt.figure(figsize=(8,6))
    plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6, label=f'Train (R²={r2_train:.2f})')
    plt.scatter(y_test, y_test_pred, color='red', alpha=0.6, label=f'Test (R²={r2_test:.2f})')
    
    # Plot garis ideal
    min_val = min(y.min(), min(y_train_pred.min(), y_test_pred.min()))
    max_val = max(y.max(), max(y_train_pred.max(), y_test_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('Experimental pActivity')
    plt.ylabel('Predicted pActivity')
    plt.title('Random Forest QSAR Model: Experimental vs Predicted') # [MODIFIKASI] Judul Plot
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_file = 'rf_results.png' # [MODIFIKASI] Nama output gambar
    plt.savefig(plot_file, dpi=300)
    print(f"      Plot disimpan ke: {plot_file}")
    
    # Menyimpan model
    model_file = 'rf_model.pkl' # [MODIFIKASI] Nama output model
    joblib.dump({'model': model, 'selector': selector}, model_file)
    print(f"      Model disimpan ke: {model_file}")

if __name__ == "__main__":
    train_model()