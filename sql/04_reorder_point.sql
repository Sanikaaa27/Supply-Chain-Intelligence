-- ============================================================
-- Q4 — Reorder Point Calculation Per Product
-- Formula: Avg Daily Sales × Avg Lead Time + Safety Stock
-- Techniques: Subqueries, arithmetic expressions, CASE WHEN
-- Purpose: Pure SQL business formula for inventory reorder alerts
-- ============================================================

USE supply_chain_intelligence;

WITH product_daily_sales AS (
    SELECT
        oi.product_id,
        DATE(o.order_purchase_timestamp)  AS sale_date,
        COUNT(oi.order_item_id)           AS units_sold,
        SUM(oi.price)                     AS revenue
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY oi.product_id, sale_date
),
product_stats AS (
    SELECT
        product_id,
        ROUND(AVG(units_sold), 2)   AS avg_daily_sales,
        ROUND(STDDEV(units_sold), 2) AS sales_stddev,
        COUNT(DISTINCT sale_date)   AS active_days
    FROM product_daily_sales
    GROUP BY product_id
),
seller_lead_avg AS (
    SELECT
        oi.product_id,
        ROUND(AVG(DATEDIFF(
            o.order_delivered_customer_date,
            o.order_purchase_timestamp
        )), 1)                          AS avg_lead_days
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
    GROUP BY oi.product_id
)
SELECT
    ps.product_id,
    p.product_category_name_english                 AS category,
    ps.avg_daily_sales,
    COALESCE(sl.avg_lead_days, 7)                   AS avg_lead_days,
    ps.sales_stddev,

    -- Safety stock = Z-score(95%) × STDDEV × SQRT(lead_time)
    ROUND(1.65 * ps.sales_stddev * SQRT(COALESCE(sl.avg_lead_days, 7)), 0)
                                                    AS safety_stock,

    -- Reorder Point = Avg Daily Sales × Lead Time + Safety Stock
    ROUND(
        ps.avg_daily_sales * COALESCE(sl.avg_lead_days, 7)
        + 1.65 * ps.sales_stddev * SQRT(COALESCE(sl.avg_lead_days, 7)),
        0
    )                                               AS reorder_point,

    CASE
        WHEN ps.avg_daily_sales > 5  THEN 'FAST_MOVER'
        WHEN ps.avg_daily_sales > 1  THEN 'REGULAR'
        ELSE 'SLOW_MOVER'
    END                                             AS velocity_class

FROM product_stats   ps
JOIN olist_products  p  ON ps.product_id = p.product_id
LEFT JOIN seller_lead_avg sl ON ps.product_id = sl.product_id
WHERE ps.active_days >= 30  -- meaningful history only
ORDER BY ps.avg_daily_sales DESC;
