## Cara Mereproduksi Environment

Proyek ini menggunakan bahasa pemrograman **Python** (direkomendasikan versi 3.9 atau lebih baru). Berikut adalah langkah-langkah untuk mengatur environment dari awal hingga siap dijalankan:

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/pataanggs/TA_Fathan
   cd TA_Fathan
   ```

2. **Buat virtual environment menggunakan `uv` (direkomendasikan):**
   Pastikan Anda telah menginstal `uv`. Jika belum, Anda bisa menginstalnya terlebih dahulu (lihat [dokumentasi resmi uv](https://docs.astral.sh/uv/getting-started/installation/)).
   ```bash
   uv venv
   # Di Windows
   .venv\Scripts\activate
   # Di Linux/Mac
   source .venv/bin/activate
   ```

3. **Instal library dan dependensi:**
   Masuk ke direktori `code` dan instal semua library yang dibutuhkan menggunakan `uv pip` agar proses instalasi jauh lebih cepat:
   ```bash
   cd code
   uv pip install -r requirements.txt
   ```

4. **Cara Menjalankan Kode Pra-pemrosesan:**
   Anda dapat menjalankan skrip yang ada di dalam folder `code/` untuk tahap pra-pemrosesan:
   ```bash
   python audio_preprocessing.py
   python text_preprocessing.py
   ```

5. **Cara Mengatur Hyperparameter dan Menjalankan Script Training:**
   Repositori ini menggunakan dua pendekatan untuk proses *fine-tuning*: **LoRA** dan **Freeze Encoder**. Sebelum menjalankan pelatihan, Anda dapat mengatur hyperparameter seperti *learning rate*, *batch size*, *epoch*, beserta parameter model spesifik pada direktorinya masing-masing.
   
   **Untuk pelatihan menggunakan metode LoRA:**
   Anda dapat mengubah hyperparameter (termasuk *rank* `r`, `lora_alpha`, dan ukuran batch) di dalam file konfigurasi (seperti `config.py` atau argumen *command prompt*). Setelah diatur, jalankan:
   ```bash
   cd lora
   # Jalankan skrip utama pelatihan (sesuaikan dengan nama file skrip Anda, misalnya train.py)
   python train.py
   ```

   **Untuk pelatihan menggunakan metode Freeze Encoder (FE):**
   Ubah hyperparameter serta konfigurasi layer mana saja yang dibekukan (*freeze*) pada file konfigurasi `freeze_encoder/config.py`. Setelah selesai diatur, jalankan:
   ```bash
   cd ../freeze_encoder
   # Jalankan skrip utama pelatihan
   python train.py
   ```

6. **Skrip Utilitas Tambahan:**
   Selain skrip utama di atas, terdapat beberapa skrip utilitas pendukung eksperimen yang juga tersedia di dalam direktori `code/`:
   *   `augmentation.py`: Digunakan untuk menerapkan augmentasi pada data latih (misalnya memberikan noise atau variasi pada audio) guna memperkaya dataset dan meningkatkan *robustness* (ketahanan) model.
   *   `metrics_logger.py`: Berfungsi untuk mencatat (*logging*) riwayat metrik penting selama proses pelatihan, seperti *Training/Validation Loss*, *Word Error Rate* (WER), dan *Character Error Rate* (CER). Ini dapat diintegrasikan dengan platform monitoring seperti Weights & Biases (W&B).
   *   `visualize_metrics.py`: Digunakan untuk membaca hasil log metrik dan memvisualisasikannya menjadi bentuk grafik/plot. Hal ini memudahkan untuk keperluan analisis apakah model mengalami *overfitting* atau *underfitting*, serta membandingkan performa antar eksperimen. Anda dapat memanggilnya melalui:
       ```bash
       cd code
       python visualize_metrics.py
       ```

## Cara Sitasi (How to Cite)

Jika Anda menggunakan kode atau referensi dari repositori ini, silakan gunakan format sitasi berikut:

**Format APA:**
> Kartagama, F. A. (2026). *Implementasi Fine Tuning Open AI Whisper Base Untuk Automatic Speech Recognition Bahasa Minangkabau* (Skripsi). Program Studi Teknik Informatika, Institut Teknologi Sumatera, Lampung.

**Format BibTeX:**
```bibtex
@mastersthesis{Kartagama2026,
  author  = {Fathan Andi Kartagama},
  title   = {Implementasi Fine Tuning Open AI Whisper Base Untuk Automatic Speech Recognition Bahasa Minangkabau},
  school  = {Institut Teknologi Sumatera},
  year    = {2026},
  type    = {Skripsi},
  address = {Lampung, Indonesia},
  note    = {Program Studi Teknik Informatika}
}
```

