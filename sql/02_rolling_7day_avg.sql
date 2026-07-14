-- ============================================================
-- Q2 — Rolling 7-Day Average Demand
-- Techniques: Window function, ROWS frame, PARTITION BY
-- Purpose: Smooth daily demand spikes to reveal true trend
-- ============================================================

USE supply_chain_intelligence;

WITH daily_demand AS (
    SELECT
        DATE(o.order_purchase_timestamp)        AS order_date,
        p.product_category_name_english         AS category,
        COUNT(oi.order_id)                      AS daily_orders,
        SUM(oi.price)                           AS daily_revenue
    FROM olist_orders       o
    JOIN olist_order_items  oi ON o.order_id    = oi.order_id
    JOIN olist_products     p  ON oi.product_id = p.product_id
    WHERE
        o.order_status = 'delivered'
        AND p.product_category_name_english IS NOT NULL
    GROUP BY
        order_date,
        category
)
SELECT
    order_date,
    category,
    daily_orders,
    daily_revenue,
    ROUND(AVG(daily_orders) OVER (
        PARTITION BY category
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                        AS rolling_7day_avg_orders,
    ROUND(AVG(daily_revenue) OVER (
        PARTITION BY category
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                        AS rolling_7day_avg_revenue
FROM daily_demand
ORDER BY
    category,
    order_date;
