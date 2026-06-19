# Prediksi Nilai Akademik Siswa (Student Grade Prediction)

Proyek ini bertujuan untuk memprediksi nilai akademik siswa berdasarkan berbagai faktor kebiasaan seperti waktu belajar, waktu bermain game, jam tidur, tingkat kehadiran, serta tingkat stres dan kecanduan. Model yang digunakan untuk melakukan prediksi adalah **CatBoost Regressor** yang dioptimasi menggunakan **Optuna**.

## 📌 Fitur Utama
- **Preprocessing Otomatis:** Melakukan kalkulasi *Study to Gaming Ratio* dan *Ordinal Encoding* pada kolom tingkat stres.
- **Seleksi Fitur (Feature Selection):** Secara otomatis menyeleksi fitur-fitur yang paling berpengaruh (membuang fitur zonk seperti *Gender*, *Gaming Genre*, *Age*).
- **Hyperparameter Tuning:** Menggunakan `Optuna` untuk mendapatkan konfigurasi model terbaik.
- **Batasan Prediksi:** Menggunakan teknik *Capping* dengan `numpy.clip()` untuk memastikan tebakan/prediksi batas nilai maksimal adalah **100**.

## 📂 Struktur File
- `data_game.csv` : Dataset utama yang berisi informasi siswa dan nilai mereka.
- `find_best_model.py` : Pipeline produksi mandiri (*Standalone Production Pipeline*). Script ini digunakan untuk melatih model, menyeleksi fitur, mencari hyperparameter terbaik dengan `Optuna`, mengevaluasi menggunakan 5-Fold Cross Validation, dan memvisualisasikan **Feature Importance**.
- `predict.py` : Script inferensi. Digunakan untuk memuat model CatBoost yang sudah dilatih (`best_catboost_model.cbm`) dan melakukan prediksi terhadap sekumpulan data, serta menyimpan hasil akhir ke dalam file `.csv`.
- `best_catboost_model.cbm` : File binary yang menyimpan model CatBoost terbaik yang dihasilkan saat fase pelatihan (dihasilkan melalui versi script training sebelumnya).

## 🚀 Cara Menjalankan

### 1. Prasyarat (Requirements)
Pastikan Python sudah terinstal di sistem Anda. Library yang dibutuhkan (seperti `pandas`, `numpy`, `catboost`, `optuna`, `scikit-learn`, `matplotlib`) akan **diinstal secara otomatis** ketika script dijalankan jika belum tersedia di sistem.

### 2. Melatih Model (Training)
Jika Anda ingin melatih ulang model dan melihat hasil evaluasinya, jalankan:
```bash
python find_best_model.py
```
*(Catatan: Script ini menggunakan `task_type="GPU"`. Jika Anda tidak menggunakan Graphic Card NVIDIA/CUDA, Anda bisa mengubah parameternya menjadi `CPU` pada baris ke-94 di dalam script).*

### 3. Melakukan Prediksi (Inference)
Untuk melakukan tebakan/prediksi terhadap dataset menggunakan model yang sudah dilatih, jalankan:
```bash
python predict.py
```
Hasil prediksi akan otomatis dibatasi maksimal 100 dan disimpan dalam file bernama `hasil_prediksi_catboost.csv`.

## 📊 Hasil Evaluasi (Contoh)
- Model yang digunakan: **CatBoost Regressor**
- Rata-rata R2 Score sangat tinggi (~100% kecocokan pada beberapa skenario) yang menunjukkan kemampuan model yang sangat baik dalam membaca pola data pada dataset `data_game.csv`.

---
*Dibuat untuk analisis kebiasaan bermain game terhadap pencapaian belajar.*
