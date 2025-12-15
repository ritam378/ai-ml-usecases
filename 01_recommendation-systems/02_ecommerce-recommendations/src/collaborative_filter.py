"""
Collaborative Filtering Recommender

This module implements collaborative filtering using matrix factorization (ALS).
For interview prep: Focus on understanding how collaborative filtering works,
not production-grade optimization.

Key Concepts:
- Matrix factorization: R ≈ U × V^T
- Implicit feedback handling
- Cold-start problem awareness
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


class CollaborativeFilter:
    """
    Collaborative filtering recommender using Non-negative Matrix Factorization (NMF).

    Interview Note: In production, you'd use libraries like `implicit` or `surprise`
    with ALS. Here we use NMF from sklearn for simplicity to demonstrate understanding.

    Key Interview Points:
    - Explain matrix factorization concept
    - Discuss handling implicit feedback
    - Address cold-start problem for new users
    """

    def __init__(self, n_factors: int = 50, max_iter: int = 200, random_state: int = 42):
        """
        Initialize collaborative filter.

        Args:
            n_factors: Number of latent factors (typical range: 50-200)
            max_iter: Maximum training iterations
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.user_factors = None  # U matrix (users × factors)
        self.item_factors = None  # V matrix (items × factors)
        self.user_mapping = {}    # Maps user_id to matrix index
        self.item_mapping = {}    # Maps product_id to matrix index
        self.reverse_user_mapping = {}  # Maps index to user_id
        self.reverse_item_mapping = {}  # Maps index to product_id

    def fit(self, interactions: pd.DataFrame) -> 'CollaborativeFilter':
        """
        Train the collaborative filtering model.

        Interview Discussion Points:
        - Why matrix factorization over simple similarity?
        - How to handle implicit feedback (views, purchases)?
        - Time-based train/test split (not random!)

        Args:
            interactions: DataFrame with columns ['user_id', 'product_id', 'confidence']
                         confidence = weight based on interaction type (view=1, purchase=5, etc.)

        Returns:
            self for method chaining
        """
        # Create user and item mappings
        unique_users = interactions['user_id'].unique()
        unique_items = interactions['product_id'].unique()

        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}

        # Build user-item interaction matrix
        user_indices = interactions['user_id'].map(self.user_mapping).values
        item_indices = interactions['product_id'].map(self.item_mapping).values
        confidences = interactions['confidence'].values

        # Create sparse matrix (memory efficient!)
        # Interview Point: Explain why sparse matrices are essential
        interaction_matrix = csr_matrix(
            (confidences, (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items))
        )

        # Train matrix factorization model
        # Interview Note: NMF ensures non-negative factors (interpretable)
        # In production, you might use ALS from `implicit` library
        self.model = NMF(
            n_components=self.n_factors,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init='nndsvd'  # Better initialization than random
        )

        # Fit and get factor matrices
        # R ≈ U × V^T (users × factors) × (factors × items)
        self.user_factors = self.model.fit_transform(interaction_matrix)  # U
        self.item_factors = self.model.components_.T  # V

        print(f"Model trained: {len(unique_users)} users, {len(unique_items)} items, {self.n_factors} factors")

        return self

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_items: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Interview Discussion:
        - What if user not in training data? (Cold-start!)
        - How to handle already-purchased items?
        - How to rank recommendations?

        Args:
            user_id: User to generate recommendations for
            n_recommendations: Number of recommendations to return
            exclude_items: Items to exclude (e.g., already purchased)

        Returns:
            List of (product_id, score) tuples, sorted by score descending
        """
        if user_id not in self.user_mapping:
            # Cold-start problem: User not in training data
            # Interview Answer: Return popular items or empty list
            print(f"Warning: User {user_id} not found (cold-start). Returning empty recommendations.")
            return []

        # Get user's latent factor vector
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_factors[user_idx]

        # Compute scores for all items: score = user_vector · item_vector
        # Interview Point: This is the matrix multiplication R ≈ U × V^T
        scores = user_vector.dot(self.item_factors.T)

        # Get top-N items
        # Interview Optimization: For large catalogs, use approximate nearest neighbors (FAISS)
        item_indices = np.argsort(scores)[::-1]  # Sort descending

        # Filter out excluded items
        recommendations = []
        for item_idx in item_indices:
            product_id = self.reverse_item_mapping[item_idx]

            # Skip if in exclude list
            if exclude_items and product_id in exclude_items:
                continue

            score = scores[item_idx]
            recommendations.append((product_id, float(score)))

            if len(recommendations) >= n_recommendations:
                break

        return recommendations

    def get_similar_items(
        self,
        product_id: str,
        n_similar: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find similar items based on item latent factors.

        Interview Use Case: "You may also like" on product pages

        Args:
            product_id: Product to find similar items for
            n_similar: Number of similar items to return

        Returns:
            List of (product_id, similarity_score) tuples
        """
        if product_id not in self.item_mapping:
            print(f"Warning: Product {product_id} not found.")
            return []

        # Get item's latent factor vector
        item_idx = self.item_mapping[product_id]
        item_vector = self.item_factors[item_idx]

        # Compute cosine similarity with all other items
        # Interview Note: Cosine similarity = dot product for normalized vectors
        # For simplicity, we use dot product here
        similarities = item_vector.dot(self.item_factors.T)

        # Get top-N similar items (excluding itself)
        similar_indices = np.argsort(similarities)[::-1]

        results = []
        for similar_idx in similar_indices:
            similar_product_id = self.reverse_item_mapping[similar_idx]

            # Skip the product itself
            if similar_product_id == product_id:
                continue

            similarity = similarities[similar_idx]
            results.append((similar_product_id, float(similarity)))

            if len(results) >= n_similar:
                break

        return results

    def get_user_item_score(self, user_id: str, product_id: str) -> float:
        """
        Get predicted score for a specific user-item pair.

        Interview Use Case: Re-ranking, scoring individual items

        Args:
            user_id: User ID
            product_id: Product ID

        Returns:
            Predicted score (higher = more likely to interact)
        """
        if user_id not in self.user_mapping or product_id not in self.item_mapping:
            return 0.0

        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[product_id]

        # Score = user_vector · item_vector
        score = self.user_factors[user_idx].dot(self.item_factors[item_idx])

        return float(score)

    def get_statistics(self) -> Dict[str, any]:
        """
        Get model statistics for monitoring/debugging.

        Interview Discussion: What metrics to monitor in production?
        """
        return {
            'n_users': len(self.user_mapping),
            'n_items': len(self.item_mapping),
            'n_factors': self.n_factors,
            'sparsity': 1.0,  # Would calculate from interaction matrix
            'model_trained': self.model is not None
        }


def create_interaction_matrix(
    interactions: pd.DataFrame,
    interaction_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Convert raw interaction logs to weighted confidence scores.

    Interview Discussion Point: How to weight different interaction types?

    Args:
        interactions: DataFrame with ['user_id', 'product_id', 'interaction_type']
        interaction_weights: Mapping of interaction type to confidence weight
                            Default: {'view': 1.0, 'cart': 3.0, 'purchase': 5.0}

    Returns:
        DataFrame with ['user_id', 'product_id', 'confidence']
    """
    if interaction_weights is None:
        # Interview Point: These weights are business-specific and should be tuned
        interaction_weights = {
            'view': 1.0,
            'cart': 3.0,
            'purchase': 5.0,
            'rating': 1.0  # Multiply by rating value if available
        }

    # Map interaction types to confidence scores
    interactions['confidence'] = interactions['interaction_type'].map(interaction_weights)

    # If rating available, multiply confidence by rating
    if 'rating' in interactions.columns:
        mask = interactions['interaction_type'] == 'rating'
        interactions.loc[mask, 'confidence'] = interactions.loc[mask, 'rating']

    # Aggregate multiple interactions for same user-item pair
    # Interview Point: User viewed product 10 times = higher confidence
    aggregated = interactions.groupby(['user_id', 'product_id']).agg({
        'confidence': 'sum',  # Sum confidences
    }).reset_index()

    return aggregated


# Example usage for interview demonstration
if __name__ == "__main__":
    # Create sample interaction data
    sample_interactions = pd.DataFrame({
        'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U3'],
        'product_id': ['P1', 'P2', 'P1', 'P2', 'P3', 'P1', 'P3', 'P4'],
        'interaction_type': ['view', 'purchase', 'view', 'cart', 'purchase', 'view', 'purchase', 'view']
    })

    # Convert to confidence scores
    interaction_matrix = create_interaction_matrix(sample_interactions)
    print("Interaction Matrix:")
    print(interaction_matrix)
    print()

    # Train model
    cf = CollaborativeFilter(n_factors=5)  # Small for demo
    cf.fit(interaction_matrix)
    print()

    # Get recommendations
    recs = cf.recommend('U1', n_recommendations=3)
    print(f"Recommendations for U1: {recs}")
    print()

    # Get similar items
    similar = cf.get_similar_items('P1', n_similar=2)
    print(f"Items similar to P1: {similar}")
