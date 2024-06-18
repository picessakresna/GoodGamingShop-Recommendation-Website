from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
df = pd.read_csv('../data-collection-preprocessing/final_product_ulasan-goodgamingshop.csv')  # Ganti dengan path ke file CSV Anda

# Create pivot table
pivot_table = df.pivot_table(index='id_user', columns='id_produk', values='rating_user').fillna(0)

# Compute the cosine similarity matrix
item_similarity = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity, index=pivot_table.columns, columns=pivot_table.columns)

def get_item_recommendations(product_id, n_recommendations=10):
    # Get similarity scores for the item
    similar_scores = item_similarity_df[product_id].sort_values(ascending=False)
    # Get top n similar items
    top_items = similar_scores.iloc[1:n_recommendations+1].index
    top_scores = similar_scores.iloc[1:n_recommendations+1].values
    
    recommendations = df[df['id_produk'].isin(top_items)][['id_produk', 'nama_produk']].drop_duplicates().reset_index(drop=True)
    recommendations['similarity_score'] = top_scores
    return recommendations.to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({"error": "Please provide a product ID"}), 400
    
    try:
        recommendations = get_item_recommendations(product_id)
        return jsonify(recommendations)
    except KeyError:
        return jsonify({"error": f"Product ID '{product_id}' not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
