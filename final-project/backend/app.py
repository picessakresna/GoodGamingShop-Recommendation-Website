from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download NLTK stop words
nltk.download('stopwords')

app = Flask(__name__)

# Function to clean text data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Function to get score by index from a list of scores
def get_score_by_idx(scores_list, idx):
    for score in scores_list:
        if score[0] == idx:
            return score[1]
    return None

# Function to load and clean data
def load_and_clean_data(products_file, reviews_file):
    df_products = pd.read_csv(products_file)
    df_reviews = pd.read_csv(reviews_file)

    # Clean product data
    df_products_cleaned = df_products.copy()
    df_products_cleaned['deskripsi'] = df_products_cleaned['deskripsi'].apply(clean_text)
    df_products_cleaned['kategori'] = df_products_cleaned['kategori'].apply(clean_text)
    df_products_cleaned.drop_duplicates(subset=['id_produk'], keep='first', inplace=True)
    df_products_cleaned['combined_features'] = df_products_cleaned['deskripsi'].fillna('') + ' ' + df_products_cleaned['kategori'].fillna('')

    return df_products, df_reviews, df_products_cleaned

# Function to create TF-IDF matrix and calculate cosine similarity
def calculate_tfidf_cosine_similarity(df):
    factory = StopWordRemoverFactory()
    stop_words_indonesia = factory.get_stop_words()
    stop_words_english = nltk.corpus.stopwords.words('english')
    combined_stop_words = stop_words_indonesia + stop_words_english

    tfidf = TfidfVectorizer(stop_words=combined_stop_words)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim_tfidf

# Function to prepare data for collaborative filtering
def prepare_collaborative_filtering_data(df_reviews):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_reviews[['id_user', 'id_produk', 'rating_user']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    return algo

# Function to create pivot table for collaborative filtering
def create_collaborative_filtering_pivot(df_reviews, df_products):
    users = df_reviews['id_user'].unique()
    products = df_products['id_produk'].unique()
    all_combinations = pd.DataFrame([(user, prod) for user in users for prod in products], columns=['id_user', 'id_produk'])
    merged_data = pd.merge(all_combinations, df_reviews, on=['id_user', 'id_produk'], how='left')
    merged_data.fillna(0, inplace=True)
    pivot_table = merged_data.pivot_table(index='id_user', columns='id_produk', values='rating_user', fill_value=0)
    cosine_sim_cf = cosine_similarity(pivot_table)

    return pivot_table, cosine_sim_cf

# Content and User-based Recommendation System Using Collaborative Filtering, Matrix Factorization, and TF-IDF Algorithms
def get_recommendations(product_ids, user_id, df_products, indices, cosine_sim_tfidf, cosine_sim_cf, algo, n_recommendations=None):
    all_recommendations = []
    for product_id in product_ids:
        try:
            idx = indices[product_id]
        except KeyError:
            return f"Product ID '{product_id}' not found.", 404
        
        if user_id not in df_reviews['id_user'].unique():  # Ensure df_reviews is accessible here
            return f"User ID '{user_id}' not found.", 404

        sim_scores_tfidf = list(enumerate(cosine_sim_tfidf[idx]))
        sim_scores_cf = list(enumerate(cosine_sim_cf[idx]))

        sim_scores_tfidf = sorted(sim_scores_tfidf, key=lambda x: x[1], reverse=True)
        sim_scores_cf = sorted(sim_scores_cf, key=lambda x: x[1], reverse=True)

        if n_recommendations is not None:
            sim_scores_tfidf = sim_scores_tfidf[1:n_recommendations+1]
            sim_scores_cf = sim_scores_cf[1:n_recommendations+1]

        combined_scores = {}
        for score in sim_scores_tfidf:
            combined_scores[score[0]] = combined_scores.get(score[0], 0) + score[1]
        for score in sim_scores_cf:
            combined_scores[score[0]] = combined_scores.get(score[0], 0) + score[1]
        
        combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        if n_recommendations is not None:
            top_indices = [i[0] for i in combined_scores[:n_recommendations]]
        else:
            top_indices = [i[0] for i in combined_scores]

        mf_scores = []
        for i in top_indices:
            if i < len(df_products):
                mf_scores.append((i, algo.predict(user_id, df_products.iloc[i]['id_produk']).est))
        
        mf_scores = sorted(mf_scores, key=lambda x: x[1], reverse=True)

        if n_recommendations is not None:
            final_indices = [i[0] for i in mf_scores[:n_recommendations]]
        else:
            final_indices = [i[0] for i in mf_scores]

        recommendations = []
        for idx in final_indices:
            product_info = df_products.iloc[idx].to_dict()
            product_info['skor_tfidf'] = get_score_by_idx(combined_scores, idx)
            product_info['skor_cf_mf'] = get_score_by_idx(mf_scores, idx)
            
            recommendations.append(product_info)

        all_recommendations.append({
            'id_produk': product_id,
            'rekomendasi': recommendations
        })
    
    # Flatten combined_recommendations into a list without id_produk structure
    flattened_recommendations = []
    seen_products = set()
    for recommendation in all_recommendations:
        for product_info in recommendation['rekomendasi']:
            product_id = product_info['id_produk']
            if product_id not in seen_products:
                flattened_recommendations.append(product_info)
                seen_products.add(product_id)
            else:
                # If product already in the list, update if the new score is higher
                for existing_product in flattened_recommendations:
                    if existing_product['id_produk'] == product_id:
                        if product_info['skor_cf_mf'] > existing_product['skor_cf_mf']:
                            existing_product.update(product_info)
                        elif product_info['skor_cf_mf'] == existing_product['skor_cf_mf']:
                            if product_info['skor_tfidf'] > existing_product['skor_tfidf']:
                                existing_product.update(product_info)
                        break

    # Sort flattened_recommendations by skor_cf_mf, skor_tfidf in descending order
    flattened_recommendations_sorted = sorted(flattened_recommendations, key=lambda x: (x['skor_cf_mf'], x['skor_tfidf']), reverse=True)

    if n_recommendations is not None:
        flattened_recommendations_sorted = flattened_recommendations_sorted[:n_recommendations]

    return flattened_recommendations_sorted

# User-Based Recommendation System Using Collaborative Filltering and Matrix Factorization Algorithms
def get_user_based_recommendations(user_id, df_products, pivot_table, algo, n_recommendations=None):
    if user_id not in pivot_table.index:
        return f"User ID '{user_id}' not found.", 404

    user_idx = pivot_table.index.get_loc(user_id)
    user_ratings = pivot_table.loc[user_id]

    # Find similar users based on cosine similarity
    similar_users = cosine_similarity(pivot_table)
    similar_users_scores = list(enumerate(similar_users[user_idx]))
    similar_users_scores = sorted(similar_users_scores, key=lambda x: x[1], reverse=True)
    
    if n_recommendations is not None:
        similar_users_scores = similar_users_scores[1:n_recommendations+1]
    else:
        similar_users_scores = similar_users_scores[1:]  # Tanpa batasan n_recommendations

    recommended_products = {}
    for user in similar_users_scores:
        similar_user_idx = user[0]
        similar_user_ratings = pivot_table.iloc[similar_user_idx]
        for product_id, rating in similar_user_ratings.items():
            if rating > 0 and product_id not in user_ratings[user_ratings > 0].index:
                if product_id in recommended_products:
                    recommended_products[product_id].append(rating)
                else:
                    recommended_products[product_id] = [rating]

    # Average the ratings and ensure they are within the 0-5 range
    for product_id in recommended_products:
        recommended_products[product_id] = min(5, sum(recommended_products[product_id]) / len(recommended_products[product_id]))

    # Predict ratings using matrix factorization
    mf_scores = []
    for product_id in df_products['id_produk'].unique():
        if product_id not in recommended_products:
            pred_rating = algo.predict(user_id, product_id).est
            mf_scores.append((product_id, pred_rating))
    
    mf_scores = sorted(mf_scores, key=lambda x: x[1], reverse=True)

    # Combine CF and MF scores
    final_recommendations = {}
    for product_id, score in recommended_products.items():
        final_recommendations[product_id] = final_recommendations.get(product_id, 0) + score
    
    for product_id, score in mf_scores:
        final_recommendations[product_id] = final_recommendations.get(product_id, 0) + score
    
    final_recommendations = sorted(final_recommendations.items(), key=lambda x: x[1], reverse=True)

    if n_recommendations is not None:
        final_recommendations = final_recommendations[:n_recommendations]

    recommendations = []
    for product_id, score in final_recommendations:
        idx = indices[product_id]
        product_info = df_products.iloc[idx].to_dict()
        product_info['skor_cf_mf'] = score  # score di sini adalah skor kombinasi CF dan MF

        recommendations.append(product_info)

    return recommendations

# Product Recommendation System that has Never Been Purchased by Users
def get_unrated_products(user_id, df_reviews, df_products, algo, n_recommendations=None):
    # Filter data for products that have never been rated by user_id
    user_reviews = df_reviews[df_reviews['id_user'] == user_id]
    rated_products = set(user_reviews['id_produk'])
    all_products = set(df_products['id_produk'])
    unrated_products = list(all_products - rated_products)
    
    # Predicted ratings for products that have not yet been rated
    predicted_ratings = []
    for product_id in unrated_products:
        pred_rating = algo.predict(user_id, product_id).est
        predicted_ratings.append((product_id, pred_rating))
    
    # Sort by highest predicted rating
    predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
    
    if n_recommendations is not None:
        top_unrated_products = predicted_ratings[:n_recommendations]
    else:
        top_unrated_products = predicted_ratings
    
    recommendations = []
    for product_id, score in top_unrated_products:
        product_info = df_products[df_products['id_produk'] == product_id].iloc[0].to_dict()
        product_info['skor_mf'] = score
        recommendations.append(product_info)

    return recommendations

# Recommendation System for Products that Have Never Been Sold
def get_products_with_zero_sales(df_products, user_id, algo, n_recommendations=None):
    products_with_zero_sales = df_products[df_products['jumlah_terjual'] == 0]['id_produk'].tolist()
    
    mf_scores = []
    for product_id in products_with_zero_sales:
        pred_rating = algo.predict(user_id, product_id).est
        mf_scores.append((product_id, pred_rating))
    
    # Sort the products by prediction score and take the best ones according to n_recommendations
    mf_scores = sorted(mf_scores, key=lambda x: x[1], reverse=True)
    
    if n_recommendations is not None:
        top_products = mf_scores[:n_recommendations]
    else:
        top_products = mf_scores
    
    recommendations = []
    for product_id, score in top_products:
        product_info = df_products[df_products['id_produk'] == product_id].iloc[0].to_dict()
        product_info['skor_mf'] = score
        recommendations.append(product_info)
    
    return recommendations

# Route to Content and User-based Recommendation System Using Collaborative Filtering, Matrix Factorization, and TF-IDF Algorithms
@app.route('/recommend', methods=['GET'])
def recommend():
    product_ids = request.args.get('product_ids')
    user_id = request.args.get('user_id')
    n_recommendations = request.args.get('n', default=None, type=int)
    if not product_ids or not user_id:
        return jsonify({"error": "Please provide both product_ids and user_id"}), 400
    
    product_ids = list(set(product_ids.split(',')))
    
    recommendations = get_recommendations(product_ids, user_id, df_products, indices, cosine_sim_tfidf, cosine_sim_cf, algo, n_recommendations)
    if isinstance(recommendations, tuple):
        return jsonify({"error": recommendations[0]}), recommendations[1]

    return jsonify(recommendations)

# Route to User-Based Recommendation System Using Collaborative Filltering and Matrix Factorization Algorithms
@app.route('/recommend_user_based', methods=['GET'])
def recommend_user_based():
    user_id = request.args.get('user_id')
    n_recommendations = request.args.get('n', default=None, type=int)

    if not user_id:
        return jsonify({"error": "Please provide user_id"}), 400

    recommendations = get_user_based_recommendations(user_id, df_products, pivot_table, algo, n_recommendations)
    if isinstance(recommendations, tuple):
        return jsonify({"error": recommendations[0]}), recommendations[1]

    return jsonify(recommendations)

# Route to Product Recommendation System that has Never Been Purchased by Users
@app.route('/unrated-products', methods=['GET'])
def unrated_products():
    user_id = request.args.get('user_id')
    n_recommendations = request.args.get('n', default=None, type=int)

    if not user_id:
        return jsonify({"error": "Please provide user_id"}), 400
    
    recommendations = get_unrated_products(user_id, df_reviews, df_products, algo, n_recommendations)
    if isinstance(recommendations, tuple):
        return jsonify({"error": recommendations[0]}), recommendations[1]
    
    return jsonify(recommendations)

# Route to Recommendation System for Products that Have Never Been Sold
@app.route('/products-with-zero-sales', methods=['GET'])
def products_with_zero_sales():
    user_id = request.args.get('user_id')
    n_recommendations = request.args.get('n', default=None, type=int)

    if not user_id:
        return jsonify({"error": "Please provide user_id"}), 400
    
    recommendations = get_products_with_zero_sales(df_products, user_id, algo, n_recommendations)
    if isinstance(recommendations, tuple):
        return jsonify({"error": recommendations[0]}), recommendations[1]
    
    return jsonify(recommendations)

# Route to get all products
@app.route('/products', methods=['GET'])
def get_all_products():
    n_recommendations = request.args.get('n', default=None, type=int)

    products = df_products.to_dict(orient='records')

    if n_recommendations is not None:
        products = df_products.head(n_recommendations).to_dict(orient='records')
    else:
        products = df_products.to_dict(orient='records')

    return jsonify(products)

# Route to get product by ID
@app.route('/products/<product_id>', methods=['GET'])
def get_product_by_id(product_id):
    product = df_products[df_products['id_produk'] == product_id]
    if product.empty:
        return jsonify({"error": f"Product ID '{product_id}' not found"}), 404

    product_info = product.iloc[0].to_dict()
    return jsonify(product_info)

# Route to get all users
@app.route('/users', methods=['GET'])
def get_all_users():
    n_recommendations = request.args.get('n', default=None, type=int)

    if n_recommendations is not None:
        users = df_reviews['id_user'].head(n_recommendations).drop_duplicates().tolist()
    else:
        users = df_reviews['id_user'].drop_duplicates().tolist()

    return jsonify(users)

# Route to get user by ID
@app.route('/users/<user_id>', methods=['GET'])
def get_user_by_id(user_id):
    user = df_reviews[df_reviews['id_user'] == user_id]
    if user.empty:
        return jsonify({"error": f"User ID '{user_id}' not found"}), 404

    user_info = user.to_dict(orient='records')
    return jsonify(user_info)

if __name__ == '__main__':
    # Load and clean data
    df_products, df_reviews, df_products_cleaned = load_and_clean_data('../data-collection-preprocessing/data-produk/clean_product-goodgamingshop.csv', '../data-collection-preprocessing/data-ulasan-clean/clean_data-ulasan-goodgamingstore.csv')

    # Calculate TF-IDF cosine similarity
    cosine_sim_tfidf = calculate_tfidf_cosine_similarity(df_products_cleaned)

    # Prepare data for collaborative filtering
    algo = prepare_collaborative_filtering_data(df_reviews)

    # Create pivot table for collaborative filtering
    pivot_table, cosine_sim_cf = create_collaborative_filtering_pivot(df_reviews, df_products)

    # Create mapping indices and ID produk
    indices = pd.Series(df_products.index, index=df_products['id_produk']).drop_duplicates()

    # Run Flask app
    app.run(debug=True)
