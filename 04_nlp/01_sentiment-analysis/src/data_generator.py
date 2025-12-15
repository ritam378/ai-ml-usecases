"""
Synthetic Data Generator for Sentiment Analysis

Generates realistic product reviews with labeled sentiments for testing and development.
"""

import random
import json
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd


class SentimentDataGenerator:
    """
    Generate synthetic product review data with sentiment labels.

    Example:
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=1000)
        generator.save_to_csv(data, "reviews.csv")
    """

    POSITIVE_TEMPLATES = [
        "This product is {adj}! Highly recommend it.",
        "{adj} quality! Exceeded my expectations.",
        "Love it! {adj} purchase, will buy again.",
        "Amazing {noun}! Works perfectly.",
        "Best {noun} I've ever bought. {adj}!",
        "Excellent {noun}, {adj} value for money.",
        "Outstanding! The {noun} is {adj}.",
        "Perfect! Exactly what I needed. {adj}!",
        "Great {noun}! Very {adj} and reliable.",
        "Fantastic! Would definitely recommend this {noun}.",
    ]

    NEGATIVE_TEMPLATES = [
        "Terrible {noun}. {adj} waste of money.",
        "Poor quality. {adj} and disappointing.",
        "Do not buy! {adj} product, broke quickly.",
        "Awful {noun}. Not worth it at all.",
        "Disappointed. {adj} quality, expected better.",
        "Worst {noun} ever. {adj} experience.",
        "Complete waste. {adj} and unreliable.",
        "Horrible! The {noun} is {adj}.",
        "Very {adj}. Returned it immediately.",
        "Terrible experience. {adj} product quality.",
    ]

    NEUTRAL_TEMPLATES = [
        "It's okay. Does what it's supposed to do.",
        "Average {noun}. Nothing special.",
        "Works as described. {adj} but not amazing.",
        "It's fine. {adj} for the price.",
        "Decent {noun}. Could be better.",
        "Not bad, not great. Just {adj}.",
        "Acceptable quality. {adj} overall.",
        "It works. {adj} but meets basic needs.",
        "Fair {noun}. {adj} performance.",
        "Mediocre. The {noun} is just {adj}.",
    ]

    POSITIVE_ADJ = [
        "excellent", "amazing", "perfect", "outstanding", "fantastic",
        "wonderful", "superb", "great", "brilliant", "exceptional",
        "impressive", "remarkable", "splendid", "terrific", "awesome"
    ]

    NEGATIVE_ADJ = [
        "terrible", "awful", "horrible", "poor", "bad",
        "disappointing", "subpar", "inferior", "defective", "faulty",
        "broken", "useless", "worthless", "cheap", "flimsy"
    ]

    NEUTRAL_ADJ = [
        "okay", "acceptable", "adequate", "fair", "decent",
        "reasonable", "standard", "average", "ordinary", "basic",
        "satisfactory", "passable", "tolerable", "moderate", "sufficient"
    ]

    NOUNS = [
        "product", "item", "purchase", "device", "gadget",
        "tool", "equipment", "appliance", "accessory", "thing"
    ]

    PRODUCT_CATEGORIES = [
        "Electronics", "Home & Kitchen", "Sports & Outdoors",
        "Books", "Clothing", "Toys & Games", "Beauty & Personal Care",
        "Health & Household", "Tools & Home Improvement", "Pet Supplies"
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize data generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.seed = seed

    def generate_review(self, sentiment: str) -> Tuple[str, str, int]:
        """
        Generate a single review with specified sentiment.

        Args:
            sentiment: Target sentiment ('positive', 'negative', 'neutral')

        Returns:
            Tuple of (review_text, sentiment_label, rating)
        """
        if sentiment == "positive":
            templates = self.POSITIVE_TEMPLATES
            adjectives = self.POSITIVE_ADJ
            rating = random.choice([4, 5])
        elif sentiment == "negative":
            templates = self.NEGATIVE_TEMPLATES
            adjectives = self.NEGATIVE_ADJ
            rating = random.choice([1, 2])
        else:  # neutral
            templates = self.NEUTRAL_TEMPLATES
            adjectives = self.NEUTRAL_ADJ
            rating = 3

        template = random.choice(templates)
        text = template.format(
            adj=random.choice(adjectives),
            noun=random.choice(self.NOUNS)
        )

        return text, sentiment, rating

    def generate(
        self,
        num_samples: int = 1000,
        distribution: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Generate multiple reviews.

        Args:
            num_samples: Number of reviews to generate
            distribution: Sentiment distribution (default: 60% pos, 25% neu, 15% neg)

        Returns:
            List of review dictionaries

        Example:
            >>> generator = SentimentDataGenerator()
            >>> data = generator.generate(num_samples=100)
            >>> len(data)
            100
        """
        if distribution is None:
            distribution = {
                "positive": 0.60,
                "neutral": 0.25,
                "negative": 0.15
            }

        # Normalize distribution
        total = sum(distribution.values())
        distribution = {k: v / total for k, v in distribution.items()}

        # Calculate counts
        counts = {
            sentiment: int(num_samples * prob)
            for sentiment, prob in distribution.items()
        }

        # Adjust for rounding
        diff = num_samples - sum(counts.values())
        if diff > 0:
            counts["positive"] += diff

        # Generate reviews
        reviews = []
        review_id = 1

        for sentiment, count in counts.items():
            for _ in range(count):
                text, label, rating = self.generate_review(sentiment)

                review = {
                    "review_id": f"REV{review_id:06d}",
                    "text": text,
                    "sentiment": label,
                    "rating": rating,
                    "product_id": f"PROD{random.randint(1, 100):03d}",
                    "product_category": random.choice(self.PRODUCT_CATEGORIES),
                    "verified_purchase": random.choice([True, False]),
                    "helpful_votes": random.randint(0, 50),
                    "review_date": self._random_date(),
                }

                reviews.append(review)
                review_id += 1

        # Shuffle to mix sentiments
        random.shuffle(reviews)

        return reviews

    def _random_date(self) -> str:
        """Generate random review date within last year."""
        start_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        review_date = start_date + timedelta(days=random_days)
        return review_date.strftime("%Y-%m-%d")

    def save_to_csv(self, data: List[Dict], filepath: str):
        """
        Save generated data to CSV file.

        Args:
            data: Generated review data
            filepath: Output CSV filepath
        """
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(data)} reviews to {filepath}")

    def save_to_json(self, data: List[Dict], filepath: str):
        """
        Save generated data to JSON file.

        Args:
            data: Generated review data
            filepath: Output JSON filepath
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} reviews to {filepath}")

    def get_statistics(self, data: List[Dict]) -> Dict:
        """
        Get statistics about generated data.

        Args:
            data: Generated review data

        Returns:
            Dictionary with statistics
        """
        df = pd.DataFrame(data)

        stats = {
            "total_reviews": len(df),
            "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
            "rating_distribution": df["rating"].value_counts().to_dict(),
            "avg_text_length": df["text"].str.len().mean(),
            "category_distribution": df["product_category"].value_counts().to_dict(),
        }

        return stats


def generate_sample_dataset(output_dir: str = "data", num_samples: int = 1000):
    """
    Generate and save sample dataset.

    Args:
        output_dir: Output directory for data files
        num_samples: Number of samples to generate
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    generator = SentimentDataGenerator()

    # Generate data
    print(f"Generating {num_samples} reviews...")
    data = generator.generate(num_samples=num_samples)

    # Save in multiple formats
    csv_path = os.path.join(output_dir, "reviews.csv")
    json_path = os.path.join(output_dir, "reviews.json")

    generator.save_to_csv(data, csv_path)
    generator.save_to_json(data, json_path)

    # Print statistics
    stats = generator.get_statistics(data)
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))

    return data


if __name__ == "__main__":
    # Generate sample dataset
    generate_sample_dataset(num_samples=1000)
