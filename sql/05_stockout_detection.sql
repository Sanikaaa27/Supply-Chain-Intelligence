-- ============================================================
-- Q5 — Stockout Detection — Demand vs. Inventory Gap
-- Techniques: Window function, gap detection, multi-table JOIN
-- Purpose: Flag consecutive low-stock days & revenue impact
-- ============================================================

USE supply_chain_intelligence;

WITH daily_category_demand AS (
    SELECT
        DATE(o.order_purchase_timestamp)         AS demand_date,
        p.product_category_name_english          AS category,
        COUNT(oi.order_item_id)                  AS units_demanded,
        SUM(oi.price)                            AS revenue_at_risk
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id    = oi.order_id
    JOIN olist_products    p  ON oi.product_id = p.product_id
    WHERE
        o.order_status = 'delivered'
        AND p.product_category_name_english IS NOT NULL
    GROUP BY demand_date, category
),
demand_with_stats AS (
    SELECT
        demand_date,
        category,
        units_demanded,
        revenue_at_risk,
        AVG(units_demanded) OVER (
            PARTITION BY category
            ORDER BY demand_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                        AS rolling_30d_avg,
        STDDEV(units_demanded) OVER (
            PARTITION BY category
            ORDER BY demand_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                        AS rolling_30d_std
    FROM daily_category_demand
),
stockout_flags AS (
    SELECT
        *,
        -- Flag as stockout if demand drops >50% below 30-day average
        CASE
            WHEN units_demanded < (rolling_30d_avg * 0.5)
             AND rolling_30d_avg IS NOT NULL
             AND rolling_30d_avg > 0
            THEN 1
            ELSE 0
        END                                      AS potential_stockout,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY demand_date
        ) - ROW_NUMBER() OVER (
            PARTITION BY category, (
                CASE
                    WHEN units_demanded < (rolling_30d_avg * 0.5)
                     AND rolling_30d_avg IS NOT NULL
                     AND rolling_30d_avg > 0
                    THEN 1 ELSE 0
                END
            )
            ORDER BY demand_date
        )                                        AS stockout_group
    FROM demand_with_stats
    WHERE rolling_30d_avg IS NOT NULL
)
SELECT
    category,
    MIN(demand_date)                             AS stockout_start,
    MAX(demand_date)                             AS stockout_end,
    COUNT(*)                                     AS consecutive_days,
    SUM(revenue_at_risk)                         AS estimated_revenue_loss,
    ROUND(AVG(rolling_30d_avg), 1)               AS expected_daily_demand,
    ROUND(AVG(units_demanded),  1)               AS actual_daily_demand
FROM stockout_flags
WHERE potential_stockout = 1
GROUP BY category, stockout_group
HAVING consecutive_days >= 3   -- only flag sustained stockouts (3+ days)
ORDER BY estimated_revenue_loss DESC;
