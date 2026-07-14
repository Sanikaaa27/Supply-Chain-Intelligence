-- ============================================================
-- Q9 — ABC Analysis — Pareto 80/20
-- Techniques: Window Functions, Running Totals, PERCENT_RANK
-- Purpose: Inventory Prioritization & Revenue Contribution
-- ============================================================

USE supply_chain_intelligence;

WITH product_revenue AS (
    SELECT
        oi.product_id,
        p.product_category_name_english AS category,

        COUNT(oi.order_item_id) AS total_units,

        ROUND(
            SUM(oi.price),
            2
        ) AS total_revenue

    FROM olist_order_items oi
    JOIN olist_products p
        ON oi.product_id = p.product_id

    WHERE p.product_category_name_english IS NOT NULL

    GROUP BY
        oi.product_id,
        category
),

ranked AS (
    SELECT
        *,

        SUM(total_revenue) OVER () AS grand_total,

        SUM(total_revenue) OVER (
            ORDER BY total_revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ) AS running_total,

        PERCENT_RANK() OVER (
            ORDER BY total_revenue DESC
        ) AS revenue_percentile,

        DENSE_RANK() OVER (
            ORDER BY total_revenue DESC
        ) AS revenue_rank

    FROM product_revenue
),

final_abc AS (
    SELECT
        *,

        ROUND(
            100.0 * total_revenue / grand_total,
            3
        ) AS revenue_share_pct,

        ROUND(
            100.0 * running_total / grand_total,
            1
        ) AS cumulative_pct

    FROM ranked
)

SELECT
    product_id,

    category,

    revenue_rank,

    total_units,

    total_revenue,

    revenue_share_pct,

    cumulative_pct,

    ROUND(
        revenue_percentile * 100,
        2
    ) AS revenue_percentile_pct,

    CASE
        WHEN cumulative_pct <= 80 THEN 'A'
        WHEN cumulative_pct <= 95 THEN 'B'
        ELSE 'C'
    END AS abc_class,

    CASE
        WHEN cumulative_pct <= 80
        THEN 'Maintain high safety stock - weekly monitoring'

        WHEN cumulative_pct <= 95
        THEN 'Moderate inventory - monthly review'

        ELSE 'Minimal stock - reorder on demand'
    END AS inventory_policy,

    CASE
        WHEN cumulative_pct <= 80 THEN 100
        WHEN cumulative_pct <= 95 THEN 60
        ELSE 20
    END AS inventory_priority_score,

    ROUND(
        AVG(total_revenue)
        OVER (
            PARTITION BY category
        ),
        2
    ) AS avg_category_revenue

FROM final_abc

ORDER BY
    total_revenue DESC;