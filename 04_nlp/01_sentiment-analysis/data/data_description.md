# Sentiment Analysis Dataset Description

## Overview

This directory contains synthetic product review data for training and testing sentiment analysis models.

## Dataset Files

- `reviews.csv` - Main dataset in CSV format (1,000 reviews)
- `reviews.json` - Main dataset in JSON format (1,000 reviews)

## Schema

### Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| review_id | string | Unique review identifier | "REV000001" |
| text | string | Review text content | "This product is excellent!" |
| sentiment | string | Sentiment label | "positive", "negative", "neutral" |
| rating | integer | Star rating (1-5) | 5 |
| product_id | string | Product identifier | "PROD042" |
| product_category | string | Product category | "Electronics" |
| verified_purchase | boolean | Whether purchase is verified | true/false |
| helpful_votes | integer | Number of helpful votes | 12 |
| review_date | string | Review date (YYYY-MM-DD) | "2024-03-15" |

## Data Distribution

### Sentiment Distribution

- **Positive**: 60% (600 reviews) - Rating: 4-5 stars
- **Neutral**: 25% (250 reviews) - Rating: 3 stars
- **Negative**: 15% (150 reviews) - Rating: 1-2 stars

## Usage Examples

```python
import pandas as pd

df = pd.read_csv('data/reviews.csv')
print(df['sentiment'].value_counts())
```

## Generation

Dataset generated using `src/data_generator.py` with seed=42 for reproducibility.
