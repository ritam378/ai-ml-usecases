-- E-commerce Database Schema for Text-to-SQL Case Study
-- This schema represents a typical e-commerce platform

-- Customers table
CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    registration_date DATE NOT NULL,
    city TEXT,
    country TEXT DEFAULT 'USA',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    status TEXT CHECK(status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')),
    total_amount DECIMAL(10,2) NOT NULL,
    shipping_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- Order Items table (many-to-many relationship between orders and products)
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Reviews table
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comment TEXT,
    review_date DATE NOT NULL,
    helpful_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- Sample Data

-- Insert customers
INSERT INTO customers (name, email, registration_date, city, country) VALUES
('John Doe', 'john.doe@email.com', '2024-01-15', 'New York', 'USA'),
('Jane Smith', 'jane.smith@email.com', '2024-02-20', 'Los Angeles', 'USA'),
('Bob Johnson', 'bob.j@email.com', '2024-03-10', 'Chicago', 'USA'),
('Alice Williams', 'alice.w@email.com', '2024-01-25', 'San Francisco', 'USA'),
('Charlie Brown', 'charlie.b@email.com', '2024-04-05', 'Seattle', 'USA'),
('Diana Prince', 'diana.p@email.com', '2024-02-14', 'Boston', 'USA'),
('Eve Anderson', 'eve.a@email.com', '2024-03-20', 'Austin', 'USA'),
('Frank Miller', 'frank.m@email.com', '2024-01-30', 'Denver', 'USA'),
('Grace Lee', 'grace.l@email.com', '2024-04-15', 'Portland', 'USA'),
('Henry Davis', 'henry.d@email.com', '2024-02-28', 'Miami', 'USA');

-- Insert products
INSERT INTO products (name, category, price, stock_quantity, description) VALUES
('Laptop Pro 15', 'Electronics', 1299.99, 50, 'Professional laptop with 15-inch display'),
('Wireless Mouse', 'Electronics', 29.99, 200, 'Ergonomic wireless mouse'),
('USB-C Cable', 'Electronics', 12.99, 500, '6ft USB-C charging cable'),
('Desk Lamp', 'Home & Office', 45.99, 100, 'Adjustable LED desk lamp'),
('Office Chair', 'Home & Office', 299.99, 30, 'Ergonomic office chair'),
('Running Shoes', 'Sports', 89.99, 150, 'Lightweight running shoes'),
('Yoga Mat', 'Sports', 24.99, 80, 'Non-slip yoga mat'),
('Water Bottle', 'Sports', 15.99, 300, 'Insulated stainless steel bottle'),
('Backpack', 'Accessories', 49.99, 120, 'Waterproof laptop backpack'),
('Sunglasses', 'Accessories', 79.99, 90, 'UV protection sunglasses'),
('Bluetooth Speaker', 'Electronics', 69.99, 75, 'Portable waterproof speaker'),
('Notebook Set', 'Home & Office', 14.99, 250, 'Set of 3 notebooks'),
('Fitness Tracker', 'Electronics', 129.99, 60, 'Heart rate and activity tracker'),
('Coffee Maker', 'Home & Office', 89.99, 40, 'Programmable coffee maker'),
('Tennis Racket', 'Sports', 119.99, 35, 'Professional tennis racket');

-- Insert orders
INSERT INTO orders (customer_id, order_date, status, total_amount, shipping_address) VALUES
(1, '2024-04-01', 'delivered', 1342.98, '123 Main St, New York, NY'),
(2, '2024-04-02', 'delivered', 89.99, '456 Oak Ave, Los Angeles, CA'),
(3, '2024-04-03', 'shipped', 374.98, '789 Pine Rd, Chicago, IL'),
(1, '2024-04-05', 'delivered', 45.99, '123 Main St, New York, NY'),
(4, '2024-04-06', 'processing', 229.97, '321 Elm St, San Francisco, CA'),
(5, '2024-04-07', 'delivered', 159.98, '654 Maple Dr, Seattle, WA'),
(2, '2024-04-08', 'cancelled', 299.99, '456 Oak Ave, Los Angeles, CA'),
(6, '2024-04-09', 'delivered', 1429.98, '987 Cedar Ln, Boston, MA'),
(7, '2024-04-10', 'shipped', 94.98, '147 Birch Ct, Austin, TX'),
(3, '2024-04-11', 'pending', 69.99, '789 Pine Rd, Chicago, IL'),
(8, '2024-04-12', 'delivered', 179.98, '258 Spruce Way, Denver, CO'),
(9, '2024-04-13', 'delivered', 249.98, '369 Willow Pl, Portland, OR'),
(10, '2024-04-14', 'shipped', 139.98, '741 Ash Blvd, Miami, FL'),
(1, '2024-04-15', 'processing', 129.99, '123 Main St, New York, NY'),
(4, '2024-04-16', 'delivered', 199.97, '321 Elm St, San Francisco, CA');

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
-- Order 1
(1, 1, 1, 1299.99),
(1, 2, 1, 29.99),
(1, 3, 1, 12.99),
-- Order 2
(2, 6, 1, 89.99),
-- Order 3
(3, 5, 1, 299.99),
(3, 4, 1, 45.99),
(3, 2, 1, 29.99),
-- Order 4
(4, 4, 1, 45.99),
-- Order 5
(5, 6, 1, 89.99),
(5, 7, 2, 24.99),
(5, 8, 2, 15.99),
(5, 9, 1, 49.99),
-- Order 6
(6, 8, 2, 15.99),
(6, 13, 1, 129.99),
-- Order 7 (cancelled)
(7, 5, 1, 299.99),
-- Order 8
(8, 1, 1, 1299.99),
(8, 13, 1, 129.99),
-- Order 9
(9, 7, 2, 24.99),
(9, 12, 3, 14.99),
-- Order 10
(10, 11, 1, 69.99),
-- Order 11
(11, 10, 1, 79.99),
(11, 9, 2, 49.99),
-- Order 12
(12, 15, 2, 119.99),
-- Order 13
(13, 6, 1, 89.99),
(13, 8, 2, 15.99),
(13, 7, 1, 24.99),
-- Order 14
(14, 13, 1, 129.99),
-- Order 15
(15, 11, 1, 69.99),
(15, 14, 1, 89.99),
(15, 12, 2, 14.99);

-- Insert reviews
INSERT INTO reviews (product_id, customer_id, rating, comment, review_date) VALUES
(1, 1, 5, 'Excellent laptop! Very fast and reliable.', '2024-04-10'),
(1, 6, 4, 'Great performance but a bit pricey.', '2024-04-18'),
(2, 1, 5, 'Perfect wireless mouse, very comfortable.', '2024-04-10'),
(4, 4, 4, 'Nice lamp, bright enough for reading.', '2024-04-12'),
(5, 3, 5, 'Best office chair I have ever owned!', '2024-04-15'),
(6, 2, 4, 'Good running shoes, very comfortable.', '2024-04-08'),
(6, 5, 5, 'Amazing shoes! Perfect for marathon training.', '2024-04-16'),
(7, 5, 3, 'Yoga mat is okay, but slides a bit.', '2024-04-16'),
(8, 4, 5, 'Keeps water cold for hours!', '2024-04-14'),
(8, 5, 5, 'Best water bottle ever!', '2024-04-16'),
(10, 8, 4, 'Stylish sunglasses, good UV protection.', '2024-04-20'),
(11, 9, 5, 'Bluetooth speaker has amazing sound quality.', '2024-04-22'),
(13, 1, 5, 'Fitness tracker is very accurate and helpful.', '2024-04-20'),
(13, 6, 5, 'Love this tracker! Motivates me to exercise.', '2024-04-21'),
(15, 9, 4, 'Good tennis racket for the price.', '2024-04-23');

-- Create indexes for better query performance
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_reviews_product_id ON reviews(product_id);
CREATE INDEX idx_reviews_customer_id ON reviews(customer_id);
