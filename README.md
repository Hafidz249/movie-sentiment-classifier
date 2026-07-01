# Movie Review Sentiment Classifier - UAS Data Mining (Kelompok 8)

Aplikasi web interaktif untuk mendeteksi, menganalisis, dan mengklasifikasikan polaritas sentimen (POSITIF/NEGATIF) dari teks ulasan film berbahasa Inggris menggunakan Natural Language Processing (NLP) dan algoritma Machine Learning (Logistic Regression).

## Anggota Kelompok 8
1. **Mohammad Hafidz Al Maaher** (NIM. 434231117)
2. **Affan Rido Harris Berliansyah** (NIM. 434231064)
3. **Muhammad Haikal Bima** (NIM. 434231111)

---

## Fitur Utama
- **NLP Pipeline**: Preprocessing teks (HTML stripping, case folding, stopwords removal, stemming).
- **TF-IDF Feature Extraction**: Mengubah kata menjadi representasi angka bernilai signifikansi frekuensi.
- **Machine Learning**: Model Logistic Regression terlatih dengan akurasi tinggi pada dataset ulasan film.
- **Visualisasi UI Premium**: Antarmuka responsif berbasis Tailwind CSS dengan efek glassmorphism, visualizer alur fasa NLP, serta indikator hasil dinamis (Hijau untuk positif, Merah untuk negatif).
- **Demo Fallback Mode**: Aplikasi dapat langsung dijalankan untuk pengujian UI tanpa model yang dilatih sebelumnya menggunakan *heuristic keyphrase classifier* bawaan.

---

## 📸 Tampilan
<img width="1908" height="1735" alt="image" src="https://github.com/user-attachments/assets/39d6d958-8ed4-43d2-9196-544cd5b918de" />
<img width="941" height="446" alt="image" src="https://github.com/user-attachments/assets/24efddd3-81c5-4ba3-b654-817a88828c5f" />

---

## Panduan Instalasi & Penggunaan

### 1. Prasyarat
Pastikan Anda sudah menginstal **Python 3.8** atau versi di atasnya pada sistem Anda.

### 2. Instalasi Dependency
Buka terminal/command prompt pada folder proyek ini, lalu jalankan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi dalam Demo Mode (Tanpa Model)
Anda dapat langsung melihat dan mencoba tampilan antarmukanya secara lokal:
```bash
python app.py
```
Buka browser dan buka alamat: **`http://localhost:5000`**

### 4. Pelatihan Model dengan Dataset Kaggle (IMDB Dataset)
Untuk melatih model Machine Learning asli:
1. Unduh dataset ulasan film IMDB dari Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Simpan file hasil unduhan dengan nama `IMDB Dataset.csv` di dalam folder `data/` di proyek ini.
3. Jalankan script pelatihan:
   ```bash
   python train.py
   ```
4. Setelah proses selesai, file model `model.pkl` dan vectorizer `vectorizer.pkl` akan otomatis terbuat di folder utama.
5. Jalankan kembali aplikasi server:
   ```bash
   python app.py
   ```
6. Banner "Demo Mode" di web akan hilang, menandakan bahwa prediksi sentimen kini telah ditenagai oleh model Machine Learning yang sesungguhnya!
