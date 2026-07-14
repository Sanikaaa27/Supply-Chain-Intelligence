-- ============================================================
-- Q13 — Top Supplier Risk Dashboard
-- Techniques:
-- CTEs, Window Functions, Composite Risk Scoring
--
-- Combines:
-- 1. Lead Time Variability
-- 2. Delay Percentage
-- 3. Revenue At Risk
-- 4. Customer Sentiment
--
-- Purpose:
-- Create a unified Supplier Risk Score
-- for Power BI Supplier Intelligence Dashboard
-- ============================================================

USE supply_chain_intelligence;

WITH lead_time_risk AS (

    SELECT
        oi.seller_id,

        ROUND(
            STDDEV(
                DATEDIFF(
                    o.order_delivered_customer_date,
                    o.order_purchase_timestamp
                )
            ),
            2
        ) AS lead_time_stddev,

        ROUND(
            100.0 *
            SUM(
                CASE
                    WHEN o.order_delivered_customer_date >
                         o.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) / COUNT(*),
            2
        ) AS delay_pct

    FROM olist_orders o
    JOIN olist_order_items oi
        ON o.order_id = oi.order_id

    WHERE
        o.order_delivered_customer_date IS NOT NULL
        AND o.order_purchase_timestamp IS NOT NULL

    GROUP BY oi.seller_id
),

revenue_risk AS (

    SELECT
        oi.seller_id,

        ROUND(
            SUM(
                CASE
                    WHEN o.order_delivered_customer_date >
                         o.order_estimated_delivery_date
                    THEN oi.price + oi.freight_value
                    ELSE 0
                END
            ),
            2
        ) AS revenue_at_risk

    FROM olist_orders o
    JOIN olist_order_items oi
        ON o.order_id = oi.order_id

    GROUP BY oi.seller_id
),

sentiment_risk AS (

    SELECT
        oi.seller_id,

        ROUND(
            AVG(r.sentiment_score),
            3
        ) AS avg_sentiment,

        ROUND(
            100.0 *
            SUM(
                CASE
                    WHEN r.review_score <= 2
                    THEN 1
                    ELSE 0
                END
            ) /
            COUNT(*),
            2
        ) AS bad_review_pct

    FROM olist_order_reviews r
    JOIN olist_orders o
        ON r.order_id = o.order_id
    JOIN olist_order_items oi
        ON o.order_id = oi.order_id

    WHERE r.sentiment_score IS NOT NULL

    GROUP BY oi.seller_id
)

SELECT

    s.seller_id,
    s.seller_state,

    ROUND(ltr.lead_time_stddev,2)
        AS lead_time_variability,

    ROUND(ltr.delay_pct,2)
        AS delay_pct,

    ROUND(rr.revenue_at_risk,2)
        AS revenue_at_risk,

    ROUND(sr.avg_sentiment,3)
        AS avg_sentiment,

    ROUND(sr.bad_review_pct,2)
        AS bad_review_pct,

    -- Composite Risk Score
    ROUND(

        (
            COALESCE(ltr.delay_pct,0) * 0.35
        )

        +

        (
            COALESCE(ltr.lead_time_stddev,0) * 1.5
        )

        +

        (
            COALESCE(sr.bad_review_pct,0) * 0.25
        )

        +

        (
            ABS(COALESCE(sr.avg_sentiment,0))
            * 100 * 0.15
        ),

        2

    ) AS supplier_risk_score,

    CASE

        WHEN
        (
            COALESCE(ltr.delay_pct,0) * 0.35
            +
            COALESCE(ltr.lead_time_stddev,0) * 1.5
            +
            COALESCE(sr.bad_review_pct,0) * 0.25
            +
            ABS(COALESCE(sr.avg_sentiment,0))
            * 100 * 0.15
        ) >= 50

        THEN 'CRITICAL'

        WHEN
        (
            COALESCE(ltr.delay_pct,0) * 0.35
            +
            COALESCE(ltr.lead_time_stddev,0) * 1.5
            +
            COALESCE(sr.bad_review_pct,0) * 0.25
            +
            ABS(COALESCE(sr.avg_sentiment,0))
            * 100 * 0.15
        ) >= 30

        THEN 'HIGH'

        WHEN
        (
            COALESCE(ltr.delay_pct,0) * 0.35
            +
            COALESCE(ltr.lead_time_stddev,0) * 1.5
            +
            COALESCE(sr.bad_review_pct,0) * 0.25
            +
            ABS(COALESCE(sr.avg_sentiment,0))
            * 100 * 0.15
        ) >= 15

        THEN 'MEDIUM'

        ELSE 'LOW'

    END AS risk_category,

    DENSE_RANK() OVER (
        ORDER BY
            (
                COALESCE(ltr.delay_pct,0) * 0.35
                +
                COALESCE(ltr.lead_time_stddev,0) * 1.5
                +
                COALESCE(sr.bad_review_pct,0) * 0.25
                +
                ABS(COALESCE(sr.avg_sentiment,0))
                * 100 * 0.15
            ) DESC
    ) AS supplier_rank

FROM olist_sellers s

LEFT JOIN lead_time_risk ltr
    ON s.seller_id = ltr.seller_id

LEFT JOIN revenue_risk rr
    ON s.seller_id = rr.seller_id

LEFT JOIN sentiment_risk sr
    ON s.seller_id = sr.seller_id

ORDER BY supplier_risk_score DESC
LIMIT 100;