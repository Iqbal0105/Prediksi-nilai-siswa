import pandas as pd
import numpy as np
import catboost as cb
import os

def main():
    print("=" * 70)
    print("                 PREDIKSI DENGAN MODEL CATBOOST")
    print("=" * 70)

    # 1. Load data
    try:
        df = pd.read_csv("data_game.csv")
        print(f"[INFO] Dataset berhasil dimuat. Ukuran: {df.shape[0]} baris, {df.shape[1]} kolom.")
    except FileNotFoundError:
        print("[ERROR] File 'data_game.csv' tidak ditemukan.")
        return

    # Simpan ID jika ada
    student_ids = None
    if "student_id" in df.columns:
        student_ids = df["student_id"].values
        df = df.drop(columns=["student_id"])

    # 2. Preprocessing & Feature Engineering
    # Sama seperti pada saat training (lihat test.py)
    df["study_to_gaming_ratio"] = df["study_hours"] / (df["gaming_hours"] + 0.1)

    stress_mapping = {"Low": 0, "Medium": 1, "High": 2}
    if "stress_level" in df.columns:
        df["stress_level"] = df["stress_level"].map(stress_mapping)

    categorical_cols = ["gender", "gaming_genre"]
    # Hanya lakukan get_dummies untuk kolom yang ada
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)

    # Pisahkan target kolom 'grades' jika ini data yang sama (untuk komparasi)
    target_col = "grades"
    if target_col in df.columns:
        y_true = df[target_col].values
        X = df.drop(columns=[target_col]).astype(float)
    else:
        y_true = None
        X = df.astype(float)

    # 3. Load model CatBoost
    model_filename = "best_catboost_model.cbm"
    if not os.path.exists(model_filename):
        print(f"[ERROR] File model '{model_filename}' tidak ditemukan.")
        return
        
    try:
        model = cb.CatBoostRegressor()
        model.load_model(model_filename)
        print(f"[SUCCESS] Model '{model_filename}' berhasil dimuat.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return

    # 4. Melakukan Prediksi
    print("[INFO] Memproses prediksi...")
    preds = model.predict(X)

    # 5. Membatasi prediksi batas atas (upper bound) = 100
    # Menggunakan numpy.clip untuk membatasi nilai maksimal
    preds_clipped = np.clip(preds, a_min=None, a_max=100)
    print("[INFO] Aturan batas atas (upper = 100) telah diterapkan pada hasil prediksi.")

    # 6. Menampilkan Hasil & Menyimpan ke CSV
    result_dict = {}
    if student_ids is not None:
        result_dict["student_id"] = student_ids
    if y_true is not None:
        result_dict["Nilai_Asli"] = y_true
        
    result_dict["Prediksi_CatBoost"] = preds
    result_dict["Prediksi_Final_Max100"] = preds_clipped

    df_result = pd.DataFrame(result_dict)

    print("\n--- Contoh Hasil Prediksi (10 Sampel Pertama) ---")
    print(df_result.head(10).round(2).to_string(index=False))

    output_file = "hasil_prediksi_catboost.csv"
    df_result.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Seluruh hasil prediksi berhasil disimpan ke '{output_file}'")
    print("=" * 70)

if __name__ == "__main__":
    main()
