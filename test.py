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
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"[SUCCESS] Berhasil menginstal '{package}'!\n")
        except Exception as e:
            print(f"[ERROR] Gagal menginstal '{package}': {e}")
            print("Silakan jalankan secara manual: pip install " + package)

# Pastikan library penting terinstall
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

# Sembunyikan log default Optuna agar output rapi
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 2. LOAD DATASET
# =====================================================================
print("=" * 70)
print("       CATBOOST REGRESSOR TUNING PIPELINE (OPTUNA & K-FOLD)")
print("=" * 70)

try:
    df = pd.read_csv("data_game.csv")
    print(f"[INFO] Dataset berhasil dimuat. Ukuran awal: {df.shape[0]} baris, {df.shape[1]} kolom.")
except FileNotFoundError:
    print("[ERROR] File 'data_game.csv' tidak ditemukan. Pastikan lokasinya benar.")
    sys.exit(1)

# Drop ID mahasiswa
if "student_id" in df.columns:
    df = df.drop(columns=["student_id"])

# =====================================================================
# 3. PREPROCESSING & FEATURE ENGINEERING
# =====================================================================
print("[INFO] Melakukan Preprocessing & Feature Engineering...")

# Pembuatan fitur rasio productivity
df["study_to_gaming_ratio"] = df["study_hours"] / (df["gaming_hours"] + 0.1)

# Ordinal Encoding untuk stress_level (Low -> 0, Medium -> 1, High -> 2)
stress_mapping = {"Low": 0, "Medium": 1, "High": 2}
if "stress_level" in df.columns:
    df["stress_level"] = df["stress_level"].map(stress_mapping)
    print("  - Ordinal Encoding pada 'stress_level' berhasil.")

# One-Hot Encoding untuk gender dan gaming_genre
categorical_cols = ["gender", "gaming_genre"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
print("  - One-Hot Encoding pada 'gender' dan 'gaming_genre' berhasil.")

# Memisahkan X (fitur) dan y (target)
target_col = "grades"
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].copy()
y = df[target_col].values

# Ubah tipe data X ke float agar kompatibel secara optimal
X = X.astype(float)

print(f"[INFO] Jumlah fitur terproses: {X.shape[1]} kolom.")
print("-" * 70)


# =====================================================================
# 4. OPTUNA OBJECTIVE FUNCTION UNTUK CATBOOST
# =====================================================================
N_SPLITS = 5
N_TRIALS = 30
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def objective(trial):
    # Search Space Hyperparameter CatBoost
    params = {
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "thread_count": 1, # Menggunakan 1 thread per model untuk kestabilan cross-validation
        "iterations": trial.suggest_int("iterations", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255)
    }
    
    scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_tr, 
            y_tr, 
            eval_set=(X_val, y_val), 
            early_stopping_rounds=20, 
            verbose=False
        )
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        
    return np.mean(scores)


# =====================================================================
# 5. RUN HYPERPARAMETER TUNING
# =====================================================================
print(f"[TUNING] Memulai pencarian hyperparameter CatBoost ({N_TRIALS} Trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

print("\n" + "=" * 25 + " HASIL OPTIMASI OPTUNA " + "=" * 25)
print(f"RMSE K-Fold Terbaik: {study.best_value:.4f}")
print("Parameter Terbaik:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("=" * 72 + "\n")


# =====================================================================
# 6. EVALUASI MODEL FINAL DENGAN K-FOLD (RMSE, MAE, R2)
# =====================================================================
print("[EVALUATION] Melakukan evaluasi final menggunakan 5-Fold Cross Validation...")

best_params = study.best_params
# Kunci statis tambahan untuk model final
best_params["loss_function"] = "RMSE"
best_params["verbose"] = 0
best_params["random_seed"] = 42
best_params["thread_count"] = -1 # Gunakan seluruh core CPU untuk pelatihan model final

final_rmse_scores = []
final_mae_scores = []
final_r2_scores = []

all_actuals = []
all_predictions = []

for train_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    fold_model = cb.CatBoostRegressor(**best_params)
    fold_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
    
    preds = fold_model.predict(X_val)
    
    all_actuals.extend(y_val)
    all_predictions.extend(preds)
    
    final_rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    final_mae_scores.append(mean_absolute_error(y_val, preds))
    final_r2_scores.append(r2_score(y_val, preds))

print("\n" + "=" * 20 + " LAPORAN METRIK MODEL FINAL (CATBOOST) " + "=" * 20)
print(f"Rata-rata RMSE K-Fold : {np.mean(final_rmse_scores):.4f}")
print(f"Rata-rata MAE K-Fold  : {np.mean(final_mae_scores):.4f}")
print(f"Rata-rata R2 Score    : {np.mean(final_r2_scores):.4f} ({np.mean(final_r2_scores)*100:.2f}% kecocokan)")
print("=" * 79 + "\n")


# =====================================================================
# 7. TABEL PEMBANDING DATA ASLI VS PREDIKSI (10 Sampel Pertama)
# =====================================================================
print("Tabel Perbandingan Data Asli vs Hasil Prediksi (10 Sampel Pertama):")
df_compare = pd.DataFrame({
    "Nilai Asli (Actual)": all_actuals,
    "Tebakan Model (Predicted)": all_predictions
})
df_compare["Selisih (Absolute Error)"] = np.abs(df_compare["Nilai Asli (Actual)"] - df_compare["Tebakan Model (Predicted)"])
df_compare = df_compare.round(2)
print(df_compare.head(10).to_string(index=False))
print("-" * 70)


# =====================================================================
# 8. LATIH & SIMPAN MODEL FINAL + FEATURE IMPORTANCE
# =====================================================================
print("[FINAL] Melatih model terbaik pada seluruh dataset...")
final_model = cb.CatBoostRegressor(**best_params)
final_model.fit(X, y, verbose=False)

# Simpan Model CatBoost
model_filename = "best_catboost_model.cbm"
final_model.save_model(model_filename)
print(f"[SUCCESS] Model terbaik berhasil disimpan ke: '{model_filename}'")

# Buat Grafik Feature Importance
importances = final_model.get_feature_importance()
feat_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_importance_df["Feature"], feat_importance_df["Importance"], color="lightcoral", edgecolor="gray")
plt.title("Tingkat Kepentingan Fitur - CatBoost Regressor Terbaik")
plt.xlabel("Skor Importance")
plt.ylabel("Fitur")
plt.tight_layout()

image_name = "feature_importance_catboost.png"
plt.savefig(image_name, dpi=300)
print(f"[SUCCESS] Grafik Feature Importance berhasil disimpan sebagai '{image_name}'")

print("\n" + "=" * 70)
print("  PROSES SELESAI DENGAN SUKSES!")
print("=" * 70)