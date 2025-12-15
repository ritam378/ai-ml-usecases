"""Generate synthetic e-commerce data for interview demonstration."""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

np.random.seed(42)

# Generate products
categories = ['Electronics', 'Books', 'Clothing', 'Home & Garden', 'Sports']
brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

products = []
for i in range(100):
    products.append({
        'product_id': f'P{i+1:03d}',
        'name': f'Product {i+1}',
        'description': f'High quality {np.random.choice(["premium", "affordable", "best-selling"])} product',
        'category': np.random.choice(categories),
        'brand': np.random.choice(brands),
        'price': round(np.random.uniform(9.99, 199.99), 2)
    })

products_df = pd.DataFrame(products)

# Generate users and interactions
users = [f'U{i+1:03d}' for i in range(100)]
interaction_types = ['view', 'cart', 'purchase']

interactions = []
for user in users:
    n_interactions = np.random.randint(5, 20)
    for _ in range(n_interactions):
        interactions.append({
            'user_id': user,
            'product_id': np.random.choice(products_df['product_id']),
            'interaction_type': np.random.choice(interaction_types, p=[0.7, 0.2, 0.1]),
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 90))
        })

interactions_df = pd.DataFrame(interactions)

# Save data
products_df.to_csv('products.csv', index=False)
interactions_df.to_csv('interactions.csv', index=False)

products_df.to_json('products.json', orient='records', indent=2)
interactions_df.to_json('interactions.json', orient='records', indent=2, date_format='iso')

print(f"Generated {len(products)} products and {len(interactions)} interactions")
print("Files saved: products.csv, interactions.csv, products.json, interactions.json")
