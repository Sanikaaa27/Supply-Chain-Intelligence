-- ============================================================
-- Q8 — Demand Seasonality Index Per Category
-- Techniques: Window AVG, division for index, CASE WHEN
-- Purpose: Identify peak seasonal categories → LSTM feature
-- Index > 130 in Nov–Dec = high seasonal demand
-- ============================================================

USE supply_chain_intelligence;

WITH monthly_category_demand AS (
    SELECT
        MONTH(o.order_purchase_timestamp)            AS month_num,
        DATE_FORMAT(o.order_purchase_timestamp, '%b') AS month_name,
        p.product_category_name_english               AS category,
        COUNT(oi.order_item_id)                       AS monthly_orders,
        SUM(oi.price)                                 AS monthly_revenue
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id    = oi.order_id
    JOIN olist_products    p  ON oi.product_id = p.product_id
    WHERE
        o.order_status = 'delivered'
        AND p.product_category_name_english IS NOT NULL
        AND YEAR(o.order_purchase_timestamp) IN (2017, 2018)
    GROUP BY month_num, month_name, category
),
annual_avg AS (
    SELECT
        category,
        AVG(monthly_orders)   AS annual_avg_orders,
        AVG(monthly_revenue)  AS annual_avg_revenue
    FROM monthly_category_demand
    GROUP BY category
)
SELECT
    mcd.month_num,
    mcd.month_name,
    mcd.category,
    mcd.monthly_orders,
    ROUND(aa.annual_avg_orders, 1)                     AS annual_avg_orders,
    -- Seasonality Index: 100 = average month, >100 = above average
    ROUND(100.0 * mcd.monthly_orders / NULLIF(aa.annual_avg_orders, 0), 1)
                                                       AS seasonality_index,
    CASE
        WHEN (100.0 * mcd.monthly_orders / NULLIF(aa.annual_avg_orders, 0)) > 130
        THEN 'HIGH_SEASON'
        WHEN (100.0 * mcd.monthly_orders / NULLIF(aa.annual_avg_orders, 0)) < 70
        THEN 'LOW_SEASON'
        ELSE 'NORMAL'
    END                                                AS season_label
FROM monthly_category_demand mcd
JOIN annual_avg              aa  ON mcd.category = aa.category
ORDER BY
    mcd.category,
    mcd.month_num;
