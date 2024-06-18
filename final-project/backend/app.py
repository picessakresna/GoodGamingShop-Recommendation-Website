from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

app = Flask(__name__)

# Load data
df_merged = pd.read_csv('../data-collection-preprocessing/final_product_ulasan-goodgamingshop.csv')  # Ganti dengan path ke file CSV Anda

# Prepare data for recommendation
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents="unicode", analyzer="word",
                      token_pattern=r"\w{1,}", ngram_range=(1, 3), stop_words="english")

rec_data = df_merged.copy()
rec_data.drop_duplicates(subset="nama_produk", keep="first", inplace=True)
rec_data.reset_index(drop=True, inplace=True)
genres = rec_data["kategori"].str.split(", | , | ,").astype(str)
tfv_matrix = tfv.fit_transform(genres)

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
rec_indices = pd.Series(rec_data.index, index=rec_data["nama_produk"]).drop_duplicates()

def give_recommendation(title, sig=sig):
    try:
        idx = rec_indices[title]
    except KeyError:
        return f"Product '{title}' not found.", 404

    sig_score = list(enumerate(sig[idx]))
    sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
    sig_score = sig_score[1:11]
    product_indices = [i[0] for i in sig_score]

    rec_dic = {
        "No": list(range(1, 11)),
        "Nama Produk": df_merged["nama_produk"].iloc[product_indices].values.tolist(),
        "Rating": df_merged["rating_user"].iloc[product_indices].values.tolist()
    }
    
    return rec_dic

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a product title"}), 400
    
    recommendation = give_recommendation(title)
    if isinstance(recommendation, tuple):
        return jsonify({"error": recommendation[0]}), recommendation[1]
    
    return jsonify(recommendation)

if __name__ == '__main__':
    app.run(debug=True)
