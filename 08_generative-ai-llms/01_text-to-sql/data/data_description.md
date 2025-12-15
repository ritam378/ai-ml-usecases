# Data Description

## Overview

This directory contains the sample e-commerce database used for the Text-to-SQL case study.

## Files

- **schema.sql** - Complete database schema with sample data
- **sample_database.db** - SQLite database file (generated from schema.sql)
- **test_queries.json** - 20 test queries with expected SQL
- **create_database.py** - Script to generate the database

## Database Schema

### Tables

#### 1. customers
Stores customer information.

**Columns:**
- `id` (INTEGER, PK) - Unique customer identifier
- `name` (TEXT) - Customer name
- `email` (TEXT, UNIQUE) - Customer email address
- `registration_date` (DATE) - When customer registered
- `city` (TEXT) - Customer city
- `country` (TEXT) - Customer country (default: USA)

**Sample Size:** 10 customers

#### 2. products
Stores product catalog.

**Columns:**
- `id` (INTEGER, PK) - Unique product identifier
- `name` (TEXT) - Product name
- `category` (TEXT) - Product category
- `price` (DECIMAL) - Product price
- `stock_quantity` (INTEGER) - Available stock
- `description` (TEXT) - Product description

**Categories:** Electronics, Home & Office, Sports, Accessories
**Sample Size:** 15 products

#### 3. orders
Stores customer orders.

**Columns:**
- `id` (INTEGER, PK) - Unique order identifier
- `customer_id` (INTEGER, FK → customers.id) - Customer who placed order
- `order_date` (DATE) - When order was placed
- `status` (TEXT) - Order status: pending, processing, shipped, delivered, cancelled
- `total_amount` (DECIMAL) - Total order value
- `shipping_address` (TEXT) - Delivery address

**Sample Size:** 15 orders

#### 4. order_items
Stores individual items within orders (many-to-many: orders ↔ products).

**Columns:**
- `id` (INTEGER, PK) - Unique identifier
- `order_id` (INTEGER, FK → orders.id) - Associated order
- `product_id` (INTEGER, FK → products.id) - Product ordered
- `quantity` (INTEGER) - Quantity ordered
- `unit_price` (DECIMAL) - Price per unit at time of order

**Sample Size:** 30 order items

#### 5. reviews
Stores product reviews from customers.

**Columns:**
- `id` (INTEGER, PK) - Unique review identifier
- `product_id` (INTEGER, FK → products.id) - Product being reviewed
- `customer_id` (INTEGER, FK → customers.id) - Customer who wrote review
- `rating` (INTEGER) - Rating from 1-5
- `comment` (TEXT) - Review text
- `review_date` (DATE) - When review was written
- `helpful_count` (INTEGER) - Number of helpful votes

**Sample Size:** 15 reviews

### Relationships

```
customers (1) ──< orders (N)
orders (1) ──< order_items (N)
products (1) ──< order_items (N)
products (1) ──< reviews (N)
customers (1) ──< reviews (N)
```

## Test Queries

The `test_queries.json` file contains 20 diverse test queries covering:

**Simple Queries (8):**
- Basic filtering
- Simple aggregations
- Sorting and limiting

**Medium Complexity (10):**
- Joins (2-3 tables)
- Aggregations with GROUP BY
- NULL handling

**Complex Queries (2):**
- Multiple joins
- Conditional aggregations

## Generating the Database

To create/recreate the database:

```bash
cd data/
python3 create_database.py
```

## Usage in Code

```python
from src.schema_manager import SchemaManager

# Initialize with sample database
schema_mgr = SchemaManager('data/sample_database.db')

# Get all tables
tables = schema_mgr.get_all_tables()
# ['customers', 'products', 'orders', 'order_items', 'reviews']
```
