# Public Opinion Sentiment Analysis Dashboard

Repositori ini berisi kode sumber untuk dashboard interaktif yang digunakan untuk menganalisis sentimen opini publik dari komentar media sosial. Proyek ini membandingkan hasil klasifikasi beberapa model pemrosesan bahasa alami (NLP) untuk dua topik utama: Koperasi Desa Merah Putih (KDMP) dan Makan Bergizi Gratis (MBG).

## Ringkasan Proyek

Dashboard ini dibangun menggunakan Streamlit untuk memvisualisasikan data komentar setelah melalui pipa data pra-pemrosesan teks dan inferensi model. Aplikasi ini dikonfigurasi menggunakan tema terang secara permanen dengan membatasi perubahan variabel CSS bawaan untuk menjaga konsistensi visual.

## Dataset

Aplikasi mendukung visualisasi multi-proyek dengan membaca dua berkas data berikut:
1. `KDMP.csv`: Data komentar Tiktok terkait program Koperasi Desa Merah Putih yang sudah dipreprocess.
2. `MBG_PROCESSED.csv`: Data komentar terkait program Makan Bergizi Gratis yang sudah dipreprocess.

Kedua dataset memiliki struktur kolom utama sebagai berikut:
- `video_id`: Identifikasi unik video asal komentar.
- `username`: Nama pengguna akun yang berkomentar.
- `clean_comment`: Teks komentar setelah tahap pembersihan.
- `sentiment`: Hasil prediksi model baseline (taufiqdp).
- `aardiiiiy_sentiment`: Hasil prediksi model kontributor Aardiiiiy.
- `agufsamudra_sentiment`: Hasil prediksi model kontributor Agufsamudra (klasifikasi biner).
- `mdhugol_sentiment`: Hasil prediksi model kontributor Mdhugol.
- `final_sentiment_voting`: Hasil konsensus akhir menggunakan metode ensemble voting (modus).

## Fitur Utama

- Pemilihan Proyek Dinamis: Pengguna dapat mengganti topik analisis antara KDMP dan MBG melalui menu drop-down di sidebar.
- Filter Range Waktu: Sinkronisasi pembaruan data secara langsung tanpa tombol submit ketika range tanggal selesai dipilih.
- KPI: Menampilkan total komentar bersih, jumlah distribusi video, pengguna unik, serta volume masing-masing kategori sentimen (positif, netral, negatif).
- Visualisasi Distribusi: Representasi persentase opini menggunakan Pie Chart dan volume data menggunakan Bar Chart.
- Komparasi Lintas Model (Benchmarking): Perbandingan performa hasil klasifikasi sentimen antar-model kontributor dalam satu grafik batang kelompok.
- Analisis Tekstual Konten: Pemetaan kata yang paling sering muncul menggunakan WordCloud dengan penyaringan kata sambung (stopwords) spesifik, disertai grafik batang 10 kata kunci teratas.
- Analisis Demografi Kontributor: Grafik batang yang menunjukkan 10 pengguna paling aktif berkomentar berdasarkan data yang difilter.

## Library yang Digunakan

- Python 3
- Streamlit
- Pandas
- Plotly Express
- WordCloud
- NLTK
- Matplotlib

## Struktur Repositori

```text
├── app.py                      # Kode utama aplikasi Streamlit
├── KDMP.csv                    # Dataset proyek KDMP
└── MBG_PROCESSED.csv           # Dataset proyek MBG