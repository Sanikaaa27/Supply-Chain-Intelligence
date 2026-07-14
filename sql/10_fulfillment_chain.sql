-- ============================================================
-- Q10 — Order Fulfillment Chain (Recursive CTE)
-- Techniques: WITH RECURSIVE, CTE chain traversal
-- Purpose: Find which fulfillment stage causes most delay
-- Note: MySQL 8.0+ required for RECURSIVE CTEs
-- ============================================================

USE supply_chain_intelligence;

-- Stage-by-stage delay breakdown (non-recursive version for MySQL 5.7 compat)
WITH order_stage_times AS (
    SELECT
        order_id,
        order_status,
        order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_date,
        order_delivered_customer_date,
        order_estimated_delivery_date,

        -- Stage 1: Purchase → Approval
        TIMESTAMPDIFF(HOUR, order_purchase_timestamp, order_approved_at)
                                                    AS approval_hours,

        -- Stage 2: Approval → Carrier Pickup
        TIMESTAMPDIFF(HOUR, order_approved_at, order_delivered_carrier_date)
                                                    AS pickup_hours,

        -- Stage 3: Carrier → Customer
        TIMESTAMPDIFF(HOUR, order_delivered_carrier_date, order_delivered_customer_date)
                                                    AS delivery_hours,

        -- Total actual vs promised
        DATEDIFF(order_delivered_customer_date, order_purchase_timestamp)
                                                    AS total_actual_days,
        DATEDIFF(order_estimated_delivery_date, order_purchase_timestamp)
                                                    AS total_promised_days

    FROM olist_orders
    WHERE
        order_delivered_customer_date IS NOT NULL
        AND order_approved_at             IS NOT NULL
        AND order_delivered_carrier_date  IS NOT NULL
),
stage_summary AS (
    SELECT
        'Purchase → Approval'   AS stage,
        ROUND(AVG(approval_hours), 1)  AS avg_hours,
        ROUND(MAX(approval_hours), 0)  AS max_hours,
        ROUND(STDDEV(approval_hours), 1) AS stddev_hours,
        COUNT(*)                         AS order_count
    FROM order_stage_times
    WHERE approval_hours BETWEEN 0 AND 720   -- exclude outliers (>30 days)

    UNION ALL

    SELECT
        'Approval → Carrier Pickup',
        ROUND(AVG(pickup_hours), 1),
        ROUND(MAX(pickup_hours), 0),
        ROUND(STDDEV(pickup_hours), 1),
        COUNT(*)
    FROM order_stage_times
    WHERE pickup_hours BETWEEN 0 AND 720

    UNION ALL

    SELECT
        'Carrier → Customer Delivery',
        ROUND(AVG(delivery_hours), 1),
        ROUND(MAX(delivery_hours), 0),
        ROUND(STDDEV(delivery_hours), 1),
        COUNT(*)
    FROM order_stage_times
    WHERE delivery_hours BETWEEN 0 AND 720
)
SELECT
    stage,
    avg_hours,
    ROUND(avg_hours / 24, 1)   AS avg_days,
    max_hours,
    stddev_hours,
    order_count,
    -- Highlight the bottleneck stage (longest average)
    CASE
        WHEN avg_hours = MAX(avg_hours) OVER () THEN '⚠ BOTTLENECK'
        ELSE ''
    END                        AS bottleneck_flag
FROM stage_summary
ORDER BY avg_hours DESC;
