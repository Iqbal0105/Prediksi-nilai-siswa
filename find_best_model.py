import os
import sys
import subprocess

# =====================================================================
# 1. AUTO-INSTALL MISSING LIBRARIES
# =====================================================================
def install_if_missing(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"[INFO] Library '{package}' tidak ditemukan. Menginstal via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"[SUCCESS] Berhasil menginstal '{package}'!\n")
        except Exception as e:
            print(f"[ERROR] Gagal menginstal '{package}': {e}")

# Pastikan library utama dan pendukung GPU tersedia
install_if_missing("numpy")
install_if_missing("pandas")
install_if_missing("matplotlib")
install_if_missing("scikit-learn", "sklearn")
install_if_missing("optuna")
install_if_missing("catboost")

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import catboost as cb

# Nonaktifkan log verbose dari Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 2. LOAD DATASET
# =====================================================================
print("=" * 75)
print("     STANDALONE PRODUCTION PIPELINE: CATBOOST REGRESSOR (GPU T4)")
print("=" * 75)

try:
    # Diubah dari "/content/data_game.csv" ke "data_game.csv" agar berjalan di lokal
    url = "data_game.csv"
    df = pd.read_csv(url)
    print(f"[INFO] Dataset berhasil dimuat. Ukuran awal: {df.shape[0]} baris.")
except FileNotFoundError:
    print("[ERROR] File 'data_game.csv' tidak ditemukan di direktori saat ini")
    sys.exit(1)


# =====================================================================
# 3. PREPROCESSING & SELECTED FEATURES ONLY
# =====================================================================
print("[INFO] Melakukan Preprocessing & Memuat 7 Fitur Terbaik...")

# Capping batas atas target nilai 'grades' maksimal 100
if "grades" in df.columns:
    df["grades"] = df["grades"].clip(upper=100)

# Pembuatan fitur rasio study_to_gaming_ratio
df["study_to_gaming_ratio"] = df["study_hours"] / (df["gaming_hours"] + 0.1)

# Ordinal Encoding untuk stress_level
stress_mapping = {"Low": 0, "Medium": 1, "High": 2}
if "stress_level" in df.columns:
    df["stress_level"] = df["stress_level"].map(stress_mapping)

# DAFTAR ELEMEN FITUR TERBAIK (Menghapus Gender, Genre Game, Umur, Sosial, Device)
fitur_pilihan = [
    "study_hours",
    "study_to_gaming_ratio",
    "sleep_hours",
    "gaming_hours",
    "attendance",
    "stress_level",
    "addiction_score"
]

target_col = "grades"
fitur_pilihan = [col for col in fitur_pilihan if col in df.columns]

# Saring data
X = df[fitur_pilihan].copy().astype(float)
y = df[target_col].values

print(f"[INFO] Menggunakan {X.shape[1]} fitur utama: {fitur_pilihan}")
print("-" * 75)


# =====================================================================
# 4. TUNING HYPERPARAMETER CATBOOST VIA OPTUNA
# =====================================================================
N_SPLITS = 5
N_TRIALS = 20
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def objective_cb(trial):
    params = {
        "loss_function": "RMSE",
        "verbose": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "iterations": 150,
        "random_seed": 42,
        # Menggunakan task_type CPU jika dijalankan lokal tanpa GPU, 
        # Atau biarkan "GPU" jika komputer ini memiliki CUDA GPU
        "task_type": "GPU"  
    }
    
    scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = cb.CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        
        # Batasi prediksi maksimal 100 saat pencarian parameter
        preds = np.clip(model.predict(X_val), None, 100)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        
    return np.mean(scores)

print(f"[TUNING] Memulai pencarian hyperparameter optimal CatBoost ({N_TRIALS} Trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective_cb, n_trials=N_TRIALS)
print(f"[SUCCESS] Tuning selesai! RMSE K-Fold Terbaik: {study.best_value:.4f}")
print("-" * 50)


# =====================================================================
# 5. EVALUASI FINAL VALIDASI SILANG (5-FOLD CV)
# =====================================================================
print("[FINAL] Mengevaluasi model CatBoost terbaik pada cross-validation...")
best_params = study.best_params

all_actuals, all_predictions = [], []
final_rmse_scores, final_mae_scores, final_r2_scores = [], [], []

for train_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    fold_model = cb.CatBoostRegressor(
        **best_params, iterations=150, task_type="GPU", verbose=0, random_seed=42
    )
    fold_model.fit(X_tr, y_tr)
    
    # CAPPING PREDIKSI OUTPUT MAKSIMAL 100
    preds = np.clip(fold_model.predict(X_val), None, 100)
    
    all_actuals.extend(y_val)
    all_predictions.extend(preds)
    final_rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    final_mae_scores.append(mean_absolute_error(y_val, preds))
    final_r2_scores.append(r2_score(y_val, preds))

print("\n" + "=" * 23 + " LAPORAN EVALUASI MODEL FINAL " + "=" * 23)
print(f"Rata-rata RMSE K-Fold : {np.mean(final_rmse_scores):.4f}")
print(f"Rata-rata MAE K-Fold  : {np.mean(final_mae_scores):.4f}")
print(f"Rata-rata R2 Score    : {np.mean(final_r2_scores):.4f} ({np.mean(final_r2_scores)*100:.2f}% Cocok)")
print("=" * 76 + "\n")


# =====================================================================
# 6. DISPLAY TABEL 10 SAMPEL & RE-TRAIN FEATURE IMPORTANCE
# =====================================================================
df_compare = pd.DataFrame({
    "Nilai Asli (Actual)": all_actuals,
    "Tebakan Model (Predicted)": all_predictions,
    "Selisih (Absolute Error)": np.abs(np.array(all_actuals) - np.array(all_predictions))
}).round(2)

print("Tabel Perbandingan Data Asli vs Hasil Prediksi CatBoost (10 Sampel Pertama):")
# Diubah dari display() ke print() agar jalan di terminal lokal
print(df_compare.head(10))
print("-" * 75)

print("[VISUALIZATION] Melatih final model pada seluruh dataset untuk Feature Importance...")
final_model = cb.CatBoostRegressor(
    **best_params, iterations=150, task_type="GPU", verbose=0, random_seed=42
)
final_model.fit(X, y)

importances = final_model.get_feature_importance()

feat_importance_df = pd.DataFrame({"Feature": fitur_pilihan, "Importance": importances}).sort_values(by="Importance", ascending=True)
plt.figure(figsize=(10, 5))
plt.barh(feat_importance_df["Feature"], feat_importance_df["Importance"], color="mediumpurple", edgecolor="gray")
plt.title("Feature Importance Kompak (CatBoost) - T4 GPU Accelerated")
plt.xlabel("Skor Importance")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

print("\n" + "=" * 75)
print("     PROSES STANDALONE PIPELINE CATBOOST SELESAI DENGAN SUKSES!")
print("=" * 75)
