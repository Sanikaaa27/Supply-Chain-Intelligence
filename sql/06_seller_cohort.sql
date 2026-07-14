-- ============================================================
-- Q6 — Cohort: Seller Reliability Over Time
-- Techniques: Self JOIN, DATE_FORMAT, cohort matrix logic
-- Purpose: Track on-time delivery % per seller cohort month
-- ============================================================

USE supply_chain_intelligence;

WITH seller_first_order AS (
    SELECT
        oi.seller_id,
        DATE_FORMAT(MIN(o.order_purchase_timestamp), '%Y-%m') AS cohort_month
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    GROUP BY oi.seller_id
),
seller_monthly_perf AS (
    SELECT
        oi.seller_id,
        DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m')           AS activity_month,
        COUNT(o.order_id)                                          AS total_orders,
        SUM(CASE
            WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date
            THEN 1 ELSE 0
        END)                                                       AS on_time_orders
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
    GROUP BY oi.seller_id, activity_month
)
SELECT
    sfo.cohort_month,
    smp.activity_month,
    -- Months since first order (0 = same month as cohort)
    PERIOD_DIFF(
        REPLACE(smp.activity_month, '-', ''),
        REPLACE(sfo.cohort_month,   '-', '')
    )                                                              AS months_since_join,
    COUNT(DISTINCT smp.seller_id)                                  AS active_sellers,
    SUM(smp.total_orders)                                          AS total_orders,
    ROUND(
        100.0 * SUM(smp.on_time_orders) / NULLIF(SUM(smp.total_orders), 0),
        1
    )                                                              AS on_time_pct
FROM seller_first_order  sfo
JOIN seller_monthly_perf smp ON sfo.seller_id = smp.seller_id
GROUP BY
    sfo.cohort_month,
    smp.activity_month,
    months_since_join
ORDER BY
    sfo.cohort_month,
    months_since_join;
