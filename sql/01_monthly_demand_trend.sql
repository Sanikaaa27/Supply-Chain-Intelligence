-- ============================================================
-- Q1 — Monthly Demand Trend Per Category
-- Techniques: DATE_FORMAT, GROUP BY, ORDER BY
-- Purpose: Foundation time-series — feeds LSTM feature engineering
-- ============================================================

USE supply_chain_intelligence;

SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS order_month,
    p.product_category_name_english                  AS category,
    COUNT(oi.order_id)                               AS total_orders,
    SUM(oi.price)                                    AS total_revenue,
    ROUND(AVG(oi.price), 2)                          AS avg_order_value
FROM olist_orders       o
JOIN olist_order_items  oi ON o.order_id   = oi.order_id
JOIN olist_products     p  ON oi.product_id = p.product_id
WHERE
    o.order_status   = 'delivered'
    AND p.product_category_name_english IS NOT NULL
    AND o.order_purchase_timestamp BETWEEN '2017-01-01' AND '2018-12-31'
GROUP BY
    order_month,
    category
ORDER BY
    order_month,
    total_revenue DESC;
