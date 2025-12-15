"""
Content-Based Recommender

This module implements content-based filtering using TF-IDF and cosine similarity.
For interview prep: Focus on understanding feature extraction and similarity computation.

Key Concepts:
- TF-IDF for text features
- Cosine similarity for item similarity
- Cold-start handling for new items
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class ContentBasedRecommender:
    """
    Content-based recommender using product features.

    Interview Key Points:
    - Handles cold-start for NEW ITEMS (advantage over collaborative filtering)
    - Explainable: "Similar to product X"
    - Requires good product metadata
    - Can create "filter bubbles" (only similar items)
    """

    def __init__(self, max_features: int = 500):
        """
        Initialize content-based recommender.

        Args:
            max_features: Maximum number of TF-IDF features to extract
        """
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.item_features = None  # Combined feature matrix
        self.item_similarity = None  # Pre-computed similarity matrix
        self.item_mapping = {}  # product_id -> index
        self.reverse_item_mapping = {}  # index -> product_id

    def fit(self, products: pd.DataFrame) -> 'ContentBasedRecommender':
        """
        Train the content-based model by extracting features.

        Interview Discussion:
        - What features to use? (text, category, price, etc.)
        - How to combine different feature types?
        - Feature normalization importance

        Args:
            products: DataFrame with columns ['product_id', 'description', 'category', 'price']

        Returns:
            self for method chaining
        """
        # Create product ID mapping
        self.item_mapping = {pid: idx for idx, pid in enumerate(products['product_id'])}
        self.reverse_item_mapping = {idx: pid for pid, idx in self.item_mapping.items()}

        # Extract text features using TF-IDF
        # Interview Point: TF-IDF downweights common words, highlights unique ones
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams and bigrams
        )

        # Combine description and category for richer features
        text_data = products['description'] + ' ' + products['category']
        text_features = self.tfidf_vectorizer.fit_transform(text_data).toarray()

        # Normalize numerical features (price)
        # Interview Point: Why normalize? Different scales affect similarity
        if 'price' in products.columns:
            scaler = StandardScaler()
            price_features = scaler.fit_transform(products[['price']])
            # Combine text and price features
            self.item_features = np.hstack([text_features, price_features])
        else:
            self.item_features = text_features

        # Pre-compute item-item similarity matrix
        # Interview Optimization: For large catalogs, compute on-demand or use FAISS
        self.item_similarity = cosine_similarity(self.item_features)

        print(f"Content model trained: {len(products)} products, {self.item_features.shape[1]} features")

        return self

    def get_similar_items(
        self,
        product_id: str,
        n_similar: int = 10,
        exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find similar items based on content features.

        Interview Use Case: "You may also like" or "Similar products"

        Args:
            product_id: Product to find similar items for
            n_similar: Number of similar items to return
            exclude_self: Whether to exclude the product itself

        Returns:
            List of (product_id, similarity_score) tuples
        """
        if product_id not in self.item_mapping:
            # Interview Point: Content-based handles new items!
            # Just need to extract features and compare
            print(f"Warning: Product {product_id} not in training data")
            return []

        product_idx = self.item_mapping[product_id]

        # Get similarity scores for this product
        similarities = self.item_similarity[product_idx]

        # Get top-N similar items
        similar_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in similar_indices:
            similar_product_id = self.reverse_item_mapping[idx]

            # Skip the product itself if requested
            if exclude_self and similar_product_id == product_id:
                continue

            similarity_score = similarities[idx]
            results.append((similar_product_id, float(similarity_score)))

            if len(results) >= n_similar:
                break

        return results

    def recommend_for_user(
        self,
        user_history: List[str],
        n_recommendations: int = 10,
        exclude_items: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Recommend items based on user's historical preferences.

        Interview Strategy: Find items similar to what user already liked

        Args:
            user_history: List of product_ids the user has interacted with
            n_recommendations: Number of recommendations to return
            exclude_items: Items to exclude (e.g., already purchased)

        Returns:
            List of (product_id, score) tuples
        """
        if not user_history:
            print("Warning: Empty user history")
            return []

        # Filter valid product IDs
        valid_history = [pid for pid in user_history if pid in self.item_mapping]
        if not valid_history:
            print("Warning: No valid products in user history")
            return []

        # Get indices of user's historical items
        history_indices = [self.item_mapping[pid] for pid in valid_history]

        # Aggregate similarity scores across all items in history
        # Interview Point: Average similarity to all liked items
        aggregated_scores = np.mean(self.item_similarity[history_indices], axis=0)

        # Rank all items by aggregated score
        ranked_indices = np.argsort(aggregated_scores)[::-1]

        # Generate recommendations
        recommendations = []
        exclude_set = set(exclude_items or []) | set(user_history)

        for idx in ranked_indices:
            product_id = self.reverse_item_mapping[idx]

            if product_id in exclude_set:
                continue

            score = aggregated_scores[idx]
            recommendations.append((product_id, float(score)))

            if len(recommendations) >= n_recommendations:
                break

        return recommendations

    def get_feature_importance(self, product_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top features for a product (for explainability).

        Interview Point: Helps explain "why" items are similar

        Args:
            product_id: Product to analyze
            top_n: Number of top features to return

        Returns:
            List of (feature_name, weight) tuples
        """
        if product_id not in self.item_mapping:
            return []

        product_idx = self.item_mapping[product_id]
        feature_vector = self.item_features[product_idx]

        # Get feature names (only for TF-IDF features)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Get top features by weight
        # Only look at text features (first len(feature_names) dimensions)
        text_features = feature_vector[:len(feature_names)]
        top_indices = np.argsort(text_features)[::-1][:top_n]

        return [(feature_names[idx], float(text_features[idx])) for idx in top_indices]


def extract_product_features(products: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to prepare product data for content-based filtering.

    Interview Discussion: Feature engineering for product recommendations

    Args:
        products: Raw product DataFrame

    Returns:
        Processed DataFrame ready for content-based model
    """
    # Ensure required columns exist
    required_cols = ['product_id', 'name', 'description', 'category']
    for col in required_cols:
        if col not in products.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean text data
    products['description'] = products['description'].fillna('')
    products['category'] = products['category'].fillna('Unknown')

    # Combine name into description for richer features
    products['description'] = products['name'] + '. ' + products['description']

    return products


# Example usage for interview demonstration
if __name__ == "__main__":
    # Create sample product data
    sample_products = pd.DataFrame({
        'product_id': ['P1', 'P2', 'P3', 'P4'],
        'name': ['Wireless Headphones', 'Bluetooth Speaker', 'USB Cable', 'Phone Case'],
        'description': [
            'High quality wireless bluetooth headphones with noise cancellation',
            'Portable bluetooth speaker with great sound quality',
            'USB-C charging cable for phones and tablets',
            'Protective phone case with grip'
        ],
        'category': ['Electronics', 'Electronics', 'Accessories', 'Accessories'],
        'price': [79.99, 49.99, 9.99, 19.99]
    })

    # Train model
    cbr = ContentBasedRecommender(max_features=20)
    cbr.fit(sample_products)
    print()

    # Find similar items
    similar = cbr.get_similar_items('P1', n_similar=2)
    print(f"Items similar to P1 (Wireless Headphones): {similar}")
    print()

    # Recommend based on user history
    recs = cbr.recommend_for_user(user_history=['P1'], n_recommendations=2)
    print(f"Recommendations for user who liked P1: {recs}")
    print()

    # Feature importance
    features = cbr.get_feature_importance('P1', top_n=5)
    print(f"Top features for P1: {features}")
