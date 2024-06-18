from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os

# Download NLTK stop words
nltk.download('stopwords')

app = Flask(__name__)

def get_image_paths(image_folder):
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def get_score_by_idx(scores_list, idx):
    for score in scores_list:
        if score[0] == idx:
            return score[1]
    return None

# Load data
df_products = pd.read_csv('../data-collection-preprocessing/data-produk/clean_product-goodgamingshop.csv')
df_reviews = pd.read_csv('../data-collection-preprocessing/data-ulasan-clean/clean_data-ulasan-goodgamingstore.csv')
df_products_cleaned = df_products.copy()

# Pembersihan data
df_products_cleaned['deskripsi'] = df_products_cleaned['deskripsi'].apply(clean_text)
df_products_cleaned['kategori'] = df_products_cleaned['kategori'].apply(clean_text)
df_products_cleaned.drop_duplicates(subset=['id_produk'], keep='first', inplace=True)

# Gabungkan deskripsi dan kategori untuk TF-IDF
df_products_cleaned['combined_features'] = df_products_cleaned['deskripsi'].fillna('') + ' ' + df_products_cleaned['kategori'].fillna('')

# Daftar stop words bahasa Indonesia
factory = StopWordRemoverFactory()
stop_words_indonesia = factory.get_stop_words()

# Daftar stop words bahasa Inggris
stop_words_english = nltk.corpus.stopwords.words('english')

# Gabungkan kedua daftar stop words
combined_stop_words = stop_words_indonesia + stop_words_english

# Buat TF-IDF vectorizer dengan daftar stop words gabungan
tfidf = TfidfVectorizer(stop_words=combined_stop_words)

# Buat matriks TF-IDF
tfidf_matrix = tfidf.fit_transform(df_products_cleaned['combined_features'])

# Hitung cosine similarity matriks untuk TF-IDF
cosine_sim_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Persiapan data untuk collaborative filtering dengan Matrix Factorization
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_reviews[['id_user', 'id_produk', 'rating_user']], reader)
trainset = data.build_full_trainset()

# Gunakan algoritma SVD dari Surprise
algo = SVD()
algo.fit(trainset)

# Buat pivot table untuk collaborative filtering
# Buat dataframe dengan semua kombinasi id_user dan id_produk
users = df_reviews['id_user'].unique()
products = df_products['id_produk'].unique()
all_combinations = pd.DataFrame([(user, prod) for user in users for prod in products], columns=['id_user', 'id_produk'])

# Gabungkan dengan df_reviews untuk memasukkan rating jika ada
merged_data = pd.merge(all_combinations, df_reviews, on=['id_user', 'id_produk'], how='left')

# Mengisi nilai NaN dengan 0
merged_data.fillna(0, inplace=True)

# Buat pivot table
pivot_table = merged_data.pivot_table(index='id_user', columns='id_produk', values='rating_user', fill_value=0)

# Hitung cosine similarity matriks untuk collaborative filtering
cosine_sim_cf = cosine_similarity(pivot_table.T, pivot_table.T)

# Buat mapping indeks dan ID produk
indices = pd.Series(df_products.index, index=df_products['id_produk']).drop_duplicates()

def get_recommendations(product_id, user_id, n_recommendations=10):
    # Dapatkan indeks produk yang sesuai dengan product_id
    try:
        idx = indices[product_id]
    except KeyError:
        return f"Product ID '{product_id}' not found.", 404

    # Dapatkan skor similarity untuk TF-IDF dan collaborative filtering
    sim_scores_tfidf = list(enumerate(cosine_sim_tfidf[idx]))
    sim_scores_cf = list(enumerate(cosine_sim_cf[idx]))

    # Urutkan produk berdasarkan skor similarity
    sim_scores_tfidf = sorted(sim_scores_tfidf, key=lambda x: x[1], reverse=True)
    sim_scores_cf = sorted(sim_scores_cf, key=lambda x: x[1], reverse=True)

    # Ambil skor produk paling mirip
    sim_scores_tfidf = sim_scores_tfidf[1:n_recommendations+1]
    sim_scores_cf = sim_scores_cf[1:n_recommendations+1]

    # Gabungkan skor (rata-rata sederhana dalam contoh ini)
    combined_scores = {}
    for score in sim_scores_tfidf:
        combined_scores[score[0]] = combined_scores.get(score[0], 0) + score[1]
    for score in sim_scores_cf:
        combined_scores[score[0]] = combined_scores.get(score[0], 0) + score[1]
    
    # Dapatkan skor produk paling mirip
    combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in combined_scores[:n_recommendations]]

    # Dapatkan prediksi dari Matrix Factorization
    mf_scores = []
    for i in top_indices:
        # Validasi i agar tidak melebihi panjang df_products
        if i < len(df_products):
            mf_scores.append((i, algo.predict(user_id, df_products.iloc[i]['id_produk']).est))
    
    mf_scores = sorted(mf_scores, key=lambda x: x[1], reverse=True)

    final_indices = [i[0] for i in mf_scores[:n_recommendations]]

    # Kembalikan produk paling mirip dengan informasi lengkap dan skor
    recommendations = []
    for idx in final_indices:
        product_info = df_products.iloc[idx].to_dict()

        product_info['tfidf_score'] = get_score_by_idx(combined_scores, idx)
        product_info['cf_score'] = get_score_by_idx(mf_scores, idx)

        image_folder = product_info['image_path']
        product_info['image_paths'] = get_image_paths(image_folder)
        
        recommendations.append(product_info)

    return recommendations

@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id')
    user_id = request.args.get('user_id')
    if not product_id or not user_id:
        return jsonify({"error": "Please provide both product_id and user_id"}), 400
    
    recommendations = get_recommendations(product_id, user_id)
    if isinstance(recommendations, tuple):
        return jsonify({"error": recommendations[0]}), recommendations[1]
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
