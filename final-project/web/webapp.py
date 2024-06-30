from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, SVD
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import requests

# Download NLTK stop words
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = '8dadea2232c8bc81e3b557f2a9e9f7a2'

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Function to normalize numerical features
def normalize_features(df):
    # Extract numerical features to be normalized
    numerical_features = ['diskon', 'jumlah_terjual', 'rating', 'rating_counter']

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

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
    df_products_cleaned['nama_produk'] = df_products_cleaned['nama_produk'].apply(clean_text)
    df_products_cleaned.drop_duplicates(subset=['id_produk'], keep='first', inplace=True)
    df_products_cleaned['combined_features'] = df_products_cleaned['kategori'].fillna('') + ' ' + df_products_cleaned['nama_produk'].fillna('') + ' ' + df_products_cleaned['deskripsi'].fillna('')
    df_products_cleaned = normalize_features(df_products_cleaned)

    return df_products, df_reviews, df_products_cleaned

# Function to calculate product numeric features scores
def calculate_product_scores(df_products_cleaned):
    # Define weights for each feature
    discount_weight = 0.2
    sales_weight = 0.1
    rating_weight = 0.6
    rating_counter_weight = 0.1

    # Initialize an empty list to store (idx, score) pairs
    scores = []

    # Iterate through each row in df_products_cleaned
    for idx, product in df_products_cleaned.iterrows():
        # Calculate the score for the current product
        total_score = (discount_weight * product['diskon'] +
                       sales_weight * product['jumlah_terjual'] +
                       rating_weight * product['rating'] +
                       rating_counter_weight * product['rating_counter'])

        # Append (idx, score) pair to scores list
        scores.append((idx, total_score))

    return scores

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
    # Deduplicate and aggregate ratings (e.g., taking the mean rating if there are duplicates)
    df_reviews_agg = df_reviews.groupby(['id_user', 'id_produk'])['rating_user'].mean().reset_index()
    
    users = df_reviews_agg['id_user'].unique()
    products = df_products['id_produk'].unique()
    
    all_combinations = pd.DataFrame([(user, prod) for user in users for prod in products], columns=['id_user', 'id_produk'])
    merged_data = pd.merge(all_combinations, df_reviews_agg, on=['id_user', 'id_produk'], how='left')
    merged_data.fillna(0, inplace=True)
    
    pivot_table = merged_data.pivot_table(index='id_user', columns='id_produk', values='rating_user', fill_value=0)
    cosine_sim_cf = cosine_similarity(pivot_table)
    
    return pivot_table, cosine_sim_cf

def get_cf_product_scores(sim_scores_cf, df_reviews, indices):
    similar_user_ids = [df_reviews['id_user'].unique()[idx] for idx, _ in sim_scores_cf]
    similar_user_reviews = df_reviews[df_reviews['id_user'].isin(similar_user_ids)]
    product_ratings = similar_user_reviews.groupby('id_produk')['rating_user'].mean()
    scaled_ratings = product_ratings / 5.0
    product_scores = [(indices[product_id], score) for product_id, score in scaled_ratings.items()]
    
    return product_scores

# Content and User-based Recommendation System Using Collaborative Filtering, Matrix Factorization, and TF-IDF Algorithms
def get_recommendations(product_ids, user_id, df_products, df_reviews, pivot_table, indices, cosine_sim_tfidf, num_scores, cosine_sim_cf, algo, n_recommendations=None):
    missing_ids = [product_id for product_id in product_ids if product_id not in indices]
    if missing_ids:
        return f"Product IDs '{', '.join(missing_ids)}' not found.", 404
    
    if user_id not in pivot_table.index:
        return f"User ID '{user_id}' not found.", 404
    else:
        user_idx = pivot_table.index.get_loc(user_id)
    
    sim_scores_cf = list(enumerate(cosine_sim_cf[user_idx]))
    sim_scores_cf = sorted(sim_scores_cf, key=lambda x: x[1], reverse=True)

    if n_recommendations is not None:
        sim_scores_cf = sim_scores_cf[1:n_recommendations+1]
    else:
        sim_scores_cf = sim_scores_cf[1:]

    sim_scores_cf_product = get_cf_product_scores(sim_scores_cf, df_reviews, indices)
    
    product_indices_to_exclude = [indices[pid] for pid in product_ids if pid in indices]
    sim_scores_cf_product = [(idx, score) for idx, score in sim_scores_cf_product if idx not in product_indices_to_exclude]
    num_scores = [(idx, score) for idx, score in num_scores if idx not in product_indices_to_exclude]

    sim_scores_cf_product = sorted(sim_scores_cf_product, key=lambda x: x[1], reverse=True)
    num_scores = sorted(num_scores, key=lambda x: x[1], reverse=True)

    if n_recommendations is not None:
        sim_scores_cf_product = sim_scores_cf_product[:n_recommendations]
        num_scores = num_scores[:n_recommendations]

    all_recommendations = []
    for product_id in product_ids:
        idx = indices[product_id]

        sim_scores_tfidf = list(enumerate(cosine_sim_tfidf[idx]))
        sim_scores_tfidf = [(idx, score) for idx, score in sim_scores_tfidf if idx not in product_indices_to_exclude]
        sim_scores_tfidf = sorted(sim_scores_tfidf, key=lambda x: x[1], reverse=True)
        
        if n_recommendations is not None:
            sim_scores_tfidf = sim_scores_tfidf[1:n_recommendations+1]
        else:
            sim_scores_tfidf = sim_scores_tfidf[1:]

        weight_tfidf = 0.52
        weight_cf = 0.28
        weight_num = 0.2

        combined_scores = {}
        for score in sim_scores_tfidf:
            combined_scores[score[0]] = combined_scores.get(score[0], 0) + score[1] * weight_tfidf
        for score in sim_scores_cf_product:
            combined_scores[score[0]] = combined_scores.get(score[0], 0) + score[1] * weight_cf
        for score in num_scores:
            combined_scores[score[0]] = combined_scores.get(score[0], 0) + score[1] * weight_num
        
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
            product_info['skor'] = get_score_by_idx(mf_scores, idx)
            
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
                        if product_info['skor'] > existing_product['skor']:
                            existing_product.update(product_info)
                        break

    # Sort flattened_recommendations by skor in descending order
    flattened_recommendations_sorted = sorted(flattened_recommendations, key=lambda x: (x['skor']), reverse=True)

    if n_recommendations is not None:
        flattened_recommendations_sorted = flattened_recommendations_sorted[:n_recommendations]

    return flattened_recommendations_sorted

# User-Based Recommendation System Using Collaborative Filltering and Matrix Factorization Algorithms
def get_user_based_recommendations(user_id, df_products, pivot_table, cosine_sim_cf, algo, n_recommendations=None):
    if user_id not in pivot_table.index:
        return f"User ID '{user_id}' not found.", 404

    user_idx = pivot_table.index.get_loc(user_id)
    user_ratings = pivot_table.loc[user_id]

    # Find similar users based on cosine similarity
    sim_scores_cf = list(enumerate(cosine_sim_cf[user_idx]))
    sim_scores_cf = sorted(sim_scores_cf, key=lambda x: x[1], reverse=True)

    if n_recommendations is not None:
        sim_scores_cf = sim_scores_cf[1:n_recommendations+1]
    else:
        sim_scores_cf = sim_scores_cf[1:]
    
    recommended_products = {}
    for user in sim_scores_cf:
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
        product_info['skor'] = score

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
        product_info['skor'] = score
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
        product_info['skor'] = score
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
    
    recommendations = get_recommendations(product_ids, user_id, df_products, df_reviews, pivot_table, indices, cosine_sim_tfidf, num_scores, cosine_sim_cf, algo, n_recommendations)
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

    recommendations = get_user_based_recommendations(user_id, df_products, pivot_table, cosine_sim_cf, algo, n_recommendations)
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

@app.route('/')
def index():
    # Fetch user data from the /users endpoint
    user_response = requests.get('http://127.0.0.1:5000/users')
    
    if user_response.status_code == 200:
        users = user_response.json()
    else:
        users = []

    return render_template('index.html', users=users)

@app.route('/home', methods=['GET'])
def home():
    # Get the user_id from the query string
    user_id = request.args.get('user_id')
    n = request.args.get('n', 3)

    s = request.args.get('s', 4)
    # Make a request to fetch recommendations
    response_recommend = requests.get(f'http://127.0.0.1:5000/recommend_user_based?user_id={user_id}&n={n}')

    response_newitems = requests.get(f'http://127.0.0.1:5000/products-with-zero-sales?user_id={user_id}&n={s}')
    
    if response_recommend.status_code == 200 & response_newitems.status_code == 200:
        recommendations_value = response_recommend.json()  # Convert JSON response to Python object
        newitems_value = response_newitems.json()  # Convert JSON response to Python object
    else:
        recommendations_value = []
        newitems_value = []

    # Render the home template and pass the recommendations data
    return render_template('home.html', user_id=user_id, recommendations=recommendations_value, newitems=newitems_value)


@app.route('/home', methods=['POST'])
def handle_login():
    username = request.form['username']
    
    if not username:
        return "Please select a user."
    
    # Redirect to home.html with user_id in query re
    return redirect(url_for('home', user_id=username))

@app.route('/all_items')
def all_items():
    user_id = request.args.get('user_id')
    page = request.args.get('page', 1, type=int)
    items_per_page = 28

    # Check if user_id has changed
    if 'user_id' in session and session['user_id'] != user_id:
        # Clear selected_ids when user_id changes
        session.pop('selected_ids', None)

    session['user_id'] = user_id

    # Fetch all items
    items_response = requests.get('http://127.0.0.1:5000/products')
    
    if items_response.status_code == 200:
        allitems_value = items_response.json()
    else:
        allitems_value = []

    total_items = len(allitems_value)
    total_pages = (total_items + items_per_page - 1) // items_per_page  # ceiling division
    start_index = (page - 1) * items_per_page
    end_index = start_index + items_per_page
    paginated_items = allitems_value[start_index:end_index]

    return render_template('all_items.html', user_id=user_id, allitems=paginated_items, page=page, total_pages=total_pages)


@app.route('/recommend_page')
def recommend_page():
    user_id = request.args.get('user_id')
    user_reviews_response = requests.get(f'http://127.0.0.1:5000/users/{user_id}')
    
    if user_reviews_response.status_code == 200:
        user_reviews = user_reviews_response.json()
    else:
        user_reviews = []

    # Fetch product details for each review
    products = []
    for review in user_reviews:
        product_response = requests.get(f'http://127.0.0.1:5000/products/{review["id_produk"]}')
        if product_response.status_code == 200:
            product = product_response.json()
            products.append(product)

    # Retrieve selected IDs from session and fetch their details
    selected_ids = session.get('selected_ids', [])
    selected_products_info = []
    for product_id in selected_ids:
        product_response = requests.get(f'http://127.0.0.1:5000/products/{product_id}')
        if product_response.status_code == 200:
            product_info = product_response.json()
            selected_products_info.append(product_info)

    if not selected_ids or len(selected_ids) == 0:
        recommended_response = requests.get(f'http://127.0.0.1:5000/recommend_user_based?user_id={user_id}&n=4')
    else:
        # Construct product_ids string from selected_ids
        product_ids_str = ','.join(selected_ids)
        recommended_response = requests.get(f'http://127.0.0.1:5000/recommend?product_ids={product_ids_str}&user_id={user_id}&n=120')

    # Handle the recommendation response
    if recommended_response.status_code == 200:
        recommendations = recommended_response.json()
    else:
        # Handle error response
        recommendations = []

    # Construct product_ids string from selected_ids
    product_ids_str = ','.join(selected_ids) if selected_ids else ''

    # Fetch kategori 3 data
    kategori_3_count = {}
    if product_ids_str:
        kategori_3_response = requests.get(f'http://127.0.0.1:5000/recommend?product_ids={product_ids_str}&user_id={user_id}&n=10')
        if kategori_3_response.status_code == 200:
            kategori_3_data = kategori_3_response.json()
            # Count occurrences of each kategori_3
            for item in kategori_3_data:
                kategori_3 = item['kategori_3']
                if kategori_3 in kategori_3_count:
                    kategori_3_count[kategori_3] += 1
                else:
                    kategori_3_count[kategori_3] = 1

    else:
        kategori_3_count = {}

    return render_template('recommend_page.html', 
                           user_id=user_id, 
                           products=products, 
                           selected_products=selected_products_info, 
                           recommendations=recommendations, 
                           kategori_3_count=kategori_3_count)

@app.route('/save_selected_ids', methods=['POST'])
def save_selected_ids():
    selected_ids = request.form.getlist('selected_ids[]')
    session['selected_ids'] = selected_ids
    return 'Selected IDs saved', 200

@app.route('/get_selected_ids') 
def get_selected_ids():
    selected_ids = session.get('selected_ids', [])
    return jsonify(selected_ids=selected_ids)

@app.route('/reset_selected_ids', methods=['POST', 'GET'])
def reset_selected_ids():
    session.pop('selected_ids', None)
    return redirect(url_for('index'))

@app.route('/daftar_belanja')
def daftar_belanja():
    user_id = request.args.get('user_id')

    # Fetch user data (assuming you have a user API)
    user_response = requests.get(f'http://127.0.0.1:5000/users/{user_id}')
    if user_response.status_code != 200:
        return f"Error fetching user data: {user_response.status_code}", user_response.status_code
    
    user_reviews = user_response.json()

    # Fetch products info for reviews
    products_info = []
    for review in user_reviews:
        product_id = review['id_produk']
        product_response = requests.get(f'http://127.0.0.1:5000/products/{product_id}')
        if product_response.status_code == 200:
            product_info = product_response.json()
            products_info.append(product_info)
    
    # Retrieve selected IDs from session and fetch their details
    selected_ids = session.get('selected_ids', [])
    selected_products_info = []
    for product_id in selected_ids:
        product_response = requests.get(f'http://127.0.0.1:5000/products/{product_id}')
        if product_response.status_code == 200:
            product_info = product_response.json()
            selected_products_info.append(product_info)

    return render_template('daftar_belanja.html', user_id=user_id, reviews=user_reviews, products=products_info, selected_products=selected_products_info)


if __name__ == '__main__':
    # Load and clean data
    df_products, df_reviews, df_products_cleaned = load_and_clean_data('./data-collection-preprocessing/data-produk/clean_product-goodgamingshop.csv', './data-collection-preprocessing/data-ulasan-clean/clean_data-ulasan-goodgamingstore.csv')

    # Calculate TF-IDF cosine similarity
    cosine_sim_tfidf = calculate_tfidf_cosine_similarity(df_products_cleaned)

    # Calculate product numeric features scores
    num_scores = calculate_product_scores(df_products_cleaned)

    # Prepare data for collaborative filtering
    algo = prepare_collaborative_filtering_data(df_reviews)

    # Create pivot table for collaborative filtering
    pivot_table, cosine_sim_cf = create_collaborative_filtering_pivot(df_reviews, df_products)

    # Create mapping indices and ID produk
    indices = pd.Series(df_products.index, index=df_products['id_produk']).drop_duplicates()

    # Run Flask app
    app.run(debug=True)