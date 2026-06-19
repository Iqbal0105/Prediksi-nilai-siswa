import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# =====================================================================
# 1. LOAD DATASET
# =====================================================================
try:
    url = "data_game.csv"
    df = pd.read_csv(url)
    print(f"Berhasil memuat data! Ukuran dataset awal: {df.shape}")
except FileNotFoundError:
    print("Error: File 'data_game.csv' tidak ditemukan. Pastikan sudah di-upload ke Colab.")
    exit()

# Drop ID mahasiswa karena sifatnya unik dan tidak mengandung pola prediktif
if "student_id" in df.columns:
    df = df.drop(columns=["student_id"])


# =====================================================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# =====================================================================
target_col = "grades"

# Buat fitur rasio produktivitas
df["study_to_gaming_ratio"] = df["study_hours"] / (df["gaming_hours"] + 0.1)

# Ambil daftar kolom terupdate
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].copy()
y = df[target_col].values

# Deteksi otomatis fitur kategorikal teks
categorical_features = []
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].astype("category")
        categorical_features.append(col)

print(f"[INFO] Total fitur yang digunakan: {X.shape[1]} kolom.")
print(f"[INFO] Fitur kategorikal terdeteksi: {categorical_features}\n")
print("-" * 70)


# =====================================================================
# 3. OBJECTIVE FUNCTION UNTUK OPTUNA TUNING
# =====================================================================
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_jobs": 1,
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=150, random_state=42)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)

    return np.mean(scores)


# =====================================================================
# 4. MENJALANKAN OPTIMASI HYPERPARAMETER
# =====================================================================
if __name__ == "__main__":
    print("Memulai pencarian hyperparameter terbaik dengan Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    print("\n" + "=" * 25 + " HASIL OPTIMASI AKHIR " + "=" * 25)
    print(f"RMSE K-Fold Terbaik Pembanding: {study.best_value:.4f}")
    print("Kombinasi Parameter Terbaik:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 72 + "\n")


    # =====================================================================
    # 5. EVALUASI METRIK LENGKAP (K-FOLD UNTUK RMSE, MAE, & R2)
    # =====================================================================
    print("Melakukan evaluasi final menggunakan K-Fold Cross Validation...")

    kf_final = KFold(n_splits=3, shuffle=True, random_state=42)
    final_rmse_scores = []
    final_mae_scores = []
    final_r2_scores = []

    # Array penampung data untuk tabel pembanding
    all_actuals = []
    all_predictions = []

    for train_idx, val_idx in kf_final.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        final_model = lgb.LGBMRegressor(
            **study.best_params, n_estimators=150, random_state=42
        )
        final_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )

        preds = final_model.predict(X_val)

        # Simpan nilai asli dan tebakan
        all_actuals.extend(y_val)
        all_predictions.extend(preds)

        final_rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        final_mae_scores.append(mean_absolute_error(y_val, preds))
        final_r2_scores.append(r2_score(y_val, preds))

    print("\n" + "=" * 23 + " LAPORAN EVALUASI MODEL FINAL " + "=" * 23)
    print(f"Rata-rata RMSE K-Fold : {np.mean(final_rmse_scores):.4f}")
    print(f"Rata-rata MAE K-Fold  : {np.mean(final_mae_scores):.4f}")
    print(f"Rata-rata R2 Score    : {np.mean(final_r2_scores):.4f} ({np.mean(final_r2_scores)*100:.2f}% cocok)")
    print("=" * 76 + "\n")


    # =====================================================================
    # 6. GRAFIK: FEATURE IMPORTANCE
    # =====================================================================
    print("Menampilkan grafik Feature Importance...")
    importance_model = lgb.LGBMRegressor(
        **study.best_params, n_estimators=150, random_state=42
    )
    importance_model.fit(X, y, categorical_feature=categorical_features)

    fig, ax = plt.subplots(figsize=(10, 5))
    lgb.plot_importance(importance_model, max_num_features=12, ignore_zero=False, title=None, ax=ax)
    plt.title("Pengaruh Fitur Terhadap Prediksi Nilai Academic (Grades)")
    plt.xlabel("Skor Importance")
    plt.ylabel("Nama Fitur/Kolom")
    plt.tight_layout()
    plt.show()

    print("\n" + "-"*70 + "\n")


    # =====================================================================
    # 7. TABEL PEMBANDING DATA ASLI VS PREDIKSI (Ambil 10 Data)
    # =====================================================================
    print("Menampilkan Tabel Perbandingan Data Asli vs Hasil Prediksi (10 Sampel Pertama):")
    
    # Buat dataframe pembanding
    df_compare = pd.DataFrame({
        "Nilai Asli (Actual)": all_actuals,
        "Tebakan Model (Predicted)": all_predictions
    })
    
    # Tambahkan kolom selisih (Error Absolut) agar terlihat seberapa jauh melesetnya
    df_compare["Selisih (Absolute Error)"] = np.abs(df_compare["Nilai Asli (Actual)"] - df_compare["Tebakan Model (Predicted)"])
    
    # Membulatkan nilai desimal ke 2 angka di belakang koma agar rapi di tabel
    df_compare = df_compare.round(2)
    
    # Mengambil 10 baris pertama menggunakan head(10)
    # Di Google Colab, memanggil variabel DataFrame langsung di akhir akan otomatis mencetak tabel HTML yang rapi
    display(df_compare.head(10))