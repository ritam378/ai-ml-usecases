"""
Hybrid Recommendation System

Combines collaborative filtering and content-based approaches.
For interview prep: Focus on understanding WHY hybrid and different combination strategies.

Key Concepts:
- Weighted combination
- Cold-start handling
- Context-aware weighting
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
from collections import Counter

from collaborative_filter import CollaborativeFilter
from content_recommender import ContentBasedRecommender


class HybridRecommender:
    """
    Hybrid recommender combining collaborative and content-based filtering.

    Interview Key Point: Explain why hybrid solves weaknesses of individual approaches:
    - Collaborative: Great for active users, but cold-start problem
    - Content-based: Handles new items, but filter bubbles
    - Hybrid: Best of both worlds!
    """

    def __init__(
        self,
        collaborative_model: CollaborativeFilter,
        content_model: ContentBasedRecommender
    ):
        """
        Initialize hybrid recommender with pre-trained models.

        Args:
            collaborative_model: Trained collaborative filtering model
            content_model: Trained content-based model
        """
        self.collaborative = collaborative_model
        self.content = content_model
        self.popularity_scores = {}  # product_id -> popularity score

    def set_popular_items(self, interactions: pd.DataFrame):
        """
        Compute popularity scores for cold-start fallback.

        Interview Point: Popularity = simple but effective baseline
        """
        item_counts = Counter(interactions['product_id'])
        total = sum(item_counts.values())

        # Normalize to 0-1 range
        self.popularity_scores = {
            pid: count / total
            for pid, count in item_counts.items()
        }

    def recommend(
        self,
        user_id: str,
        user_history: Optional[List[str]] = None,
        n_recommendations: int = 10,
        context: str = 'homepage',
        exclude_items: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate hybrid recommendations.

        Interview Discussion: How to combine different signals?

        Args:
            user_id: User to recommend for
            user_history: User's product history (for content-based)
            n_recommendations: Number of recommendations
            context: Context ('homepage', 'product_page', 'cart')
            exclude_items: Items to exclude

        Returns:
            List of recommendation dicts with product_id, score, reason
        """
        # Get context-aware weights
        # Interview Point: Different contexts need different strategies
        weights = self._get_context_weights(user_id, context, user_history)

        # Collect recommendations from each approach
        collab_recs = {}
        content_recs = {}
        popular_recs = {}

        # Collaborative filtering recommendations
        if weights['collaborative'] > 0:
            try:
                cf_results = self.collaborative.recommend(
                    user_id,
                    n_recommendations=n_recommendations * 2,  # Get more candidates
                    exclude_items=exclude_items
                )
                collab_recs = {pid: score for pid, score in cf_results}
            except:
                pass  # User might not be in training data

        # Content-based recommendations
        if weights['content'] > 0 and user_history:
            try:
                cb_results = self.content.recommend_for_user(
                    user_history,
                    n_recommendations=n_recommendations * 2,
                    exclude_items=exclude_items
                )
                content_recs = {pid: score for pid, score in cb_results}
            except:
                pass

        # Popular items (fallback)
        if weights['popularity'] > 0:
            popular_recs = self.popularity_scores.copy()
            # Remove excluded items
            if exclude_items:
                for pid in exclude_items:
                    popular_recs.pop(pid, None)

        # Combine scores with weights
        # Interview Point: This is weighted linear combination
        combined_scores = {}
        all_products = set(collab_recs.keys()) | set(content_recs.keys()) | set(popular_recs.keys())

        for product_id in all_products:
            # Normalize scores to 0-1 range first (important!)
            cf_score = self._normalize_score(collab_recs.get(product_id, 0), collab_recs)
            cb_score = self._normalize_score(content_recs.get(product_id, 0), content_recs)
            pop_score = popular_recs.get(product_id, 0)

            # Weighted combination
            final_score = (
                weights['collaborative'] * cf_score +
                weights['content'] * cb_score +
                weights['popularity'] * pop_score
            )

            combined_scores[product_id] = final_score

        # Rank and return top-N
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:n_recommendations]

        # Add explanations
        # Interview Point: Explainability helps users trust recommendations
        recommendations = []
        for product_id, score in ranked:
            reason = self._generate_explanation(
                product_id,
                collab_recs,
                content_recs,
                popular_recs
            )

            recommendations.append({
                'product_id': product_id,
                'score': float(score),
                'reason': reason
            })

        return recommendations

    def _get_context_weights(
        self,
        user_id: str,
        context: str,
        user_history: Optional[List[str]]
    ) -> Dict[str, float]:
        """
        Get weights based on context and user activity.

        Interview Discussion: This is a key design decision!
        """
        # Check if user is in collaborative model (has history)
        is_new_user = user_id not in self.collaborative.user_mapping
        has_history = user_history and len(user_history) > 0

        if context == 'homepage':
            if is_new_user or not has_history:
                # Cold-start: Favor popularity
                return {'collaborative': 0.1, 'content': 0.3, 'popularity': 0.6}
            else:
                # Active user: Favor personalization
                return {'collaborative': 0.6, 'content': 0.3, 'popularity': 0.1}

        elif context == 'product_page':
            # Show similar products (content-based)
            return {'collaborative': 0.2, 'content': 0.7, 'popularity': 0.1}

        elif context == 'cart':
            # Show complementary products (collaborative)
            return {'collaborative': 0.8, 'content': 0.1, 'popularity': 0.1}

        else:
            # Default balanced weights
            return {'collaborative': 0.4, 'content': 0.4, 'popularity': 0.2}

    def _normalize_score(self, score: float, all_scores: Dict) -> float:
        """Normalize score to 0-1 range."""
        if not all_scores:
            return 0.0

        values = list(all_scores.values())
        min_val, max_val = min(values), max(values)

        if max_val == min_val:
            return 0.5

        return (score - min_val) / (max_val - min_val)

    def _generate_explanation(
        self,
        product_id: str,
        collab_scores: Dict,
        content_scores: Dict,
        popular_scores: Dict
    ) -> str:
        """Generate explanation for why this item was recommended."""
        # Check which component contributed most
        if product_id in collab_scores and collab_scores[product_id] > 0:
            return "Popular among users like you"
        elif product_id in content_scores and content_scores[product_id] > 0:
            return "Similar to items you liked"
        elif product_id in popular_scores:
            return "Trending product"
        else:
            return "Recommended for you"


# Example usage for interview demonstration
if __name__ == "__main__":
    print("Hybrid Recommender Example")
    print("Interview Note: This demonstrates combining multiple recommendation strategies")
    print()

    # In a real interview, you'd show:
    # 1. How each component works independently
    # 2. How hybrid combination improves results
    # 3. How it handles cold-start gracefully

    print("Key Interview Points:")
    print("1. Hybrid solves cold-start for both users AND items")
    print("2. Context-aware weighting adapts to different use cases")
    print("3. Weighted combination is simple but effective")
    print("4. Explainability builds user trust")
