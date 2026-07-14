-- ============================================================
-- SUPPLY CHAIN INTELLIGENCE — MySQL Schema
-- Column names match EXACT Kaggle CSV headers
-- Run this FRESH: drops and recreates the entire database
-- ============================================================

DROP DATABASE IF EXISTS supply_chain_intelligence;
CREATE DATABASE supply_chain_intelligence
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE supply_chain_intelligence;

-- ── 1. Customers ─────────────────────────────────────────────
CREATE TABLE olist_customers (
    customer_id                  VARCHAR(50) NOT NULL,
    customer_unique_id           VARCHAR(50) NOT NULL,
    customer_zip_code_prefix     INT,
    customer_city                VARCHAR(100),
    customer_state               CHAR(2),
    PRIMARY KEY (customer_id),
    INDEX idx_customer_unique (customer_unique_id),
    INDEX idx_customer_state  (customer_state)
);

-- ── 2. Geolocation ───────────────────────────────────────────
CREATE TABLE olist_geolocation (
    geolocation_zip_code_prefix  INT,
    geolocation_lat              DECIMAL(10,7),
    geolocation_lng              DECIMAL(10,7),
    geolocation_city             VARCHAR(100),
    geolocation_state            CHAR(2),
    INDEX idx_geo_zip   (geolocation_zip_code_prefix),
    INDEX idx_geo_state (geolocation_state)
);

-- ── 3. Sellers ───────────────────────────────────────────────
CREATE TABLE olist_sellers (
    seller_id                    VARCHAR(50) NOT NULL,
    seller_zip_code_prefix       INT,
    seller_city                  VARCHAR(100),
    seller_state                 CHAR(2),
    PRIMARY KEY (seller_id),
    INDEX idx_seller_state (seller_state)
);

-- ── 4. Category Translation ──────────────────────────────────
CREATE TABLE product_category_translation (
    product_category_name         VARCHAR(100) NOT NULL,
    product_category_name_english VARCHAR(100),
    PRIMARY KEY (product_category_name)
);

-- ── 5. Products ──────────────────────────────────────────────
-- Note: Kaggle CSV has typo 'lenght' (missing 't') — kept as-is
CREATE TABLE olist_products (
    product_id                    VARCHAR(50)  NOT NULL,
    product_category_name         VARCHAR(100),
    product_name_lenght           INT,
    product_description_lenght    INT,
    product_photos_qty            INT,
    product_weight_g              INT,
    product_length_cm             INT,
    product_height_cm             INT,
    product_width_cm              INT,
    product_category_name_english VARCHAR(100),
    PRIMARY KEY (product_id),
    INDEX idx_product_category (product_category_name_english)
);

-- ── 6. Orders ────────────────────────────────────────────────
CREATE TABLE olist_orders (
    order_id                       VARCHAR(50) NOT NULL,
    customer_id                    VARCHAR(50) NOT NULL,
    order_status                   VARCHAR(20),
    order_purchase_timestamp       DATETIME,
    order_approved_at              DATETIME,
    order_delivered_carrier_date   DATETIME,
    order_delivered_customer_date  DATETIME,
    order_estimated_delivery_date  DATETIME,
    PRIMARY KEY (order_id),
    INDEX idx_order_customer  (customer_id),
    INDEX idx_order_status    (order_status),
    INDEX idx_order_purchase  (order_purchase_timestamp),
    INDEX idx_order_delivered (order_delivered_customer_date),
    CONSTRAINT fk_order_customer FOREIGN KEY (customer_id)
        REFERENCES olist_customers (customer_id)
);

-- ── 7. Order Items ───────────────────────────────────────────
CREATE TABLE olist_order_items (
    order_id             VARCHAR(50)   NOT NULL,
    order_item_id        INT           NOT NULL,
    product_id           VARCHAR(50),
    seller_id            VARCHAR(50),
    shipping_limit_date  DATETIME,
    price                DECIMAL(10,2),
    freight_value        DECIMAL(10,2),
    PRIMARY KEY (order_id, order_item_id),
    INDEX idx_item_product (product_id),
    INDEX idx_item_seller  (seller_id),
    INDEX idx_seller_date  (seller_id, shipping_limit_date),
    CONSTRAINT fk_item_order   FOREIGN KEY (order_id)   REFERENCES olist_orders   (order_id),
    CONSTRAINT fk_item_product FOREIGN KEY (product_id) REFERENCES olist_products (product_id),
    CONSTRAINT fk_item_seller  FOREIGN KEY (seller_id)  REFERENCES olist_sellers  (seller_id)
);

-- ── 8. Order Payments ────────────────────────────────────────
CREATE TABLE olist_order_payments (
    order_id              VARCHAR(50)   NOT NULL,
    payment_sequential    INT,
    payment_type          VARCHAR(30),
    payment_installments  INT,
    payment_value         DECIMAL(10,2),
    INDEX idx_payment_order (order_id),
    INDEX idx_payment_type  (payment_type),
    CONSTRAINT fk_payment_order FOREIGN KEY (order_id)
        REFERENCES olist_orders (order_id)
);

-- ── 9. Order Reviews ─────────────────────────────────────────
CREATE TABLE olist_order_reviews (
    review_id                VARCHAR(50) NOT NULL,
    order_id                 VARCHAR(50) NOT NULL,
    review_score             TINYINT,
    review_comment_title     TEXT,
    review_comment_message   TEXT,
    review_creation_date     DATETIME,
    review_answer_timestamp  DATETIME,
    sentiment_score          DECIMAL(5,4) DEFAULT NULL,
    PRIMARY KEY (review_id),
    INDEX idx_review_order (order_id),
    INDEX idx_review_score (review_score),
    CONSTRAINT fk_review_order FOREIGN KEY (order_id)
        REFERENCES olist_orders (order_id)
);

-- ── Verify all 9 tables created ──────────────────────────────
SELECT table_name, table_rows
FROM information_schema.tables
WHERE table_schema = 'supply_chain_intelligence'
ORDER BY table_name;