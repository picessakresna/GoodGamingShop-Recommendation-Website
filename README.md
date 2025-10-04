# GoodGamingShop Recommendation System

## 📋 Deskripsi Proyek

GoodGamingShop Recommendation System adalah sistem rekomendasi produk gaming yang dikembangkan menggunakan berbagai algoritma machine learning untuk memberikan rekomendasi produk yang personal dan akurat kepada pengguna. Sistem ini menggunakan data dari toko gaming GoodGamingShop di Tokopedia.

## 🎯 Fitur Utama

### Sistem Rekomendasi

-   **Content-Based Filtering**: Menggunakan TF-IDF untuk analisis kategori produk
-   **Collaborative Filtering**: Menggunakan cosine similarity berdasarkan perilaku pengguna
-   **Matrix Factorization**: Menggunakan algoritma SVD untuk prediksi rating
-   **Hybrid Recommendation**: Kombinasi dari ketiga metode di atas dengan pembobotan optimal

### Jenis Rekomendasi

1. **Rekomendasi Berdasarkan Produk**: Rekomendasi produk serupa berdasarkan produk yang dipilih
2. **Rekomendasi Berdasarkan User**: Rekomendasi berdasarkan preferensi dan histori pengguna
3. **Rekomendasi Produk Belum Pernah Dibeli**: Produk yang belum pernah dibeli oleh pengguna
4. **Rekomendasi Produk Baru**: Produk dengan penjualan nol yang berpotensi diminati

### Web Interface

-   Dashboard pengguna yang intuitif
-   Halaman katalog produk dengan pagination
-   Sistem keranjang belanja virtual
-   Visualisasi rekomendasi dengan kategori produk

## 🛠️ Teknologi yang Digunakan

### Backend

-   **Python 3.x**
-   **Flask**: Web framework
-   **Pandas**: Data manipulation
-   **Scikit-learn**: Machine learning algorithms
-   **Surprise**: Collaborative filtering library
-   **NLTK & Sastrawi**: Natural language processing

### Frontend

-   **HTML5/CSS3**
-   **JavaScript**
-   **Bootstrap**: UI framework

### Data Processing

-   **Selenium**: Web scraping
-   **Jupyter Notebook**: Data analysis
-   **CSV**: Data storage format

## 📁 Struktur Proyek

```
GoodGamingShop-Recommendation-Website/
├── final-project/
│   ├── data-analysis/
│   │   └── analysis-goodgamingshop1 (1).ipynb
│   ├── data-collection-preprocessing/
│   │   ├── data-produk/
│   │   │   ├── clean_product-goodgamingshop.csv
│   │   │   ├── clean_product-goodgamingshop2.csv
│   │   │   └── product-goodgamingshop.csv
│   │   ├── data-ulasan-clean/
│   │   │   ├── clean_data-ulasan-goodgamingstore.csv
│   │   │   └── merge-data-ulasan-goodgamingstore.csv
│   │   ├── cleandata-product.ipynb
│   │   ├── cleandata-review.ipynb
│   │   ├── merge-ulasan.ipynb
│   │   ├── scrap-tokopedia-product.py
│   │   ├── scrap-tokopedia-product-gambar.py
│   │   └── scrap-tokopedia-review.py
│   ├── timeline/
│   │   └── Book.xlsx
│   └── web/
│       ├── webapp.py
│       ├── static/
│       │   ├── css/
│       │   └── images/
│       └── templates/
│           ├── index.html
│           ├── home.html
│           ├── all_items.html
│           ├── recommend_page.html
│           └── daftar_belanja.html
├── .gitignore
└── README.md
```

## 🚀 Instalasi dan Setup

### 1. Clone Repository

```bash
git clone https://github.com/picessakresna/GoodGamingShop-Recommendation-Website.git
cd GoodGamingShop-Recommendation-Website
```

### 2. Install Dependencies

```bash
pip install flask pandas scikit-learn surprise nltk sastrawi selenium requests
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
```

### 4. Setup Data

Pastikan file data berada di lokasi yang benar:

-   `final-project/data-collection-preprocessing/data-produk/clean_product-goodgamingshop.csv`
-   `final-project/data-collection-preprocessing/data-ulasan-clean/clean_data-ulasan-goodgamingstore.csv`

### 5. Jalankan Aplikasi

```bash
cd final-project/web
python webapp.py
```

Aplikasi akan berjalan di `http://127.0.0.1:5000`

## 📊 API Endpoints

### Produk

-   `GET /products` - Mendapatkan semua produk
-   `GET /products/<product_id>` - Mendapatkan produk berdasarkan ID

### Pengguna

-   `GET /users` - Mendapatkan semua pengguna
-   `GET /users/<user_id>` - Mendapatkan data pengguna berdasarkan ID

### Rekomendasi

-   `GET /recommend?product_ids=<ids>&user_id=<id>&n=<limit>` - Rekomendasi hybrid
-   `GET /recommend_user_based?user_id=<id>&n=<limit>` - Rekomendasi berbasis pengguna
-   `GET /unrated-products?user_id=<id>&n=<limit>` - Produk belum pernah dibeli
-   `GET /products-with-zero-sales?user_id=<id>&n=<limit>` - Produk dengan penjualan nol

### Parameter

-   `product_ids`: ID produk (comma-separated untuk multiple)
-   `user_id`: ID pengguna
-   `n`: Jumlah rekomendasi (opsional)

## 🔍 Cara Penggunaan

### 1. Login

-   Akses halaman utama di `http://127.0.0.1:5000`
-   Pilih pengguna dari dropdown
-   Klik "Login"

### 2. Dashboard Utama

-   Lihat rekomendasi berdasarkan preferensi pengguna
-   Jelajahi produk baru yang mungkin menarik

### 3. Katalog Produk

-   Klik "All Items" untuk melihat semua produk
-   Pilih produk yang diminati untuk keranjang

### 4. Halaman Rekomendasi

-   Klik "Get Recommendation" untuk mendapatkan rekomendasi
-   Sistem akan menampilkan produk yang direkomendasikan berdasarkan pilihan

### 5. Daftar Belanja

-   Lihat histori pembelian dan produk yang dipilih
-   Kelola keranjang belanja virtual

## 🧮 Algoritma Rekomendasi

### 1. Content-Based Filtering (52% weight)

-   Menggunakan TF-IDF Vectorizer untuk analisis kategori produk
-   Menghitung cosine similarity antar produk
-   Menyaring stop words Bahasa Indonesia dan Inggris

### 2. Collaborative Filtering (28% weight)

-   Membuat user-item matrix dari rating pengguna
-   Menghitung cosine similarity antar pengguna
-   Merekomendasikan produk berdasarkan pengguna serupa

### 3. Numerical Features (20% weight)

-   Normalisasi fitur numerik: diskon, jumlah terjual, rating, rating counter
-   Pembobotan: rating (70%), diskon (10%), penjualan (10%), rating counter (10%)
-   Menggunakan MinMaxScaler untuk normalisasi

### 4. Matrix Factorization

-   Implementasi algoritma SVD (Singular Value Decomposition)
-   Prediksi rating untuk produk yang belum dinilai pengguna
-   Digunakan sebagai filter final untuk ranking produk

## 📈 Evaluasi Sistem

Sistem menggunakan kombinasi metrik evaluasi:

-   **Precision**: Relevansi rekomendasi
-   **Recall**: Coverage rekomendasi
-   **RMSE**: Akurasi prediksi rating
-   **User Engagement**: Interaksi pengguna dengan rekomendasi

## 🔧 Konfigurasi

### Pembobotan Algoritma

```python
weight_tfidf = 0.52      # Content-based filtering
weight_cf = 0.28         # Collaborative filtering
weight_num = 0.2         # Numerical features
```

### Pembobotan Fitur Numerik

```python
discount_weight = 0.1        # Bobot diskon
sales_weight = 0.1          # Bobot penjualan
rating_weight = 0.7         # Bobot rating
rating_counter_weight = 0.1  # Bobot jumlah rating
```

## 📝 Data Collection

### Web Scraping

Proyek ini menggunakan Selenium untuk scraping data dari Tokopedia:

1. **scrap-tokopedia-product.py**: Mengumpulkan data produk

    - Nama, kategori, harga, diskon, rating, deskripsi
    - Otomatis menjelajahi semua halaman produk

2. **scrap-tokopedia-review.py**: Mengumpulkan data ulasan

    - Rating pengguna, komentar, tanggal review

3. **scrap-tokopedia-product-gambar.py**: Mengunduh gambar produk

### Data Preprocessing

-   **cleandata-product.ipynb**: Pembersihan data produk
-   **cleandata-review.ipynb**: Pembersihan data ulasan
-   **merge-ulasan.ipynb**: Penggabungan data ulasan

## 🚦 Troubleshooting

### Error Umum

1. **ModuleNotFoundError**: Pastikan semua dependencies terinstall
2. **FileNotFoundError**: Periksa path file CSV data
3. **Memory Error**: Reduce ukuran dataset untuk testing
4. **Port sudah digunakan**: Ubah port di `app.run(port=5001)`

### Optimasi Performance

-   Gunakan caching untuk hasil rekomendasi
-   Implement lazy loading untuk data besar
-   Optimalkan query database untuk response time

## 🤝 Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 Lisensi

Proyek ini dikembangkan untuk tujuan edukasi dan penelitian.

## 🙏 Acknowledgments

-   Tokopedia untuk sumber data
-   GoodGamingShop sebagai studi kasus
-   Komunitas open source untuk libraries yang digunakan

---

**Note**: Pastikan untuk mematuhi terms of service dan robot.txt saat melakukan web scraping. Gunakan data secara bertanggung jawab dan untuk tujuan edukasi.
