-- ============================================================
-- Q11 — Review Sentiment → Supply Failure Early Warning System
-- Techniques: CTEs, Multi-table JOINs, Risk Scoring
-- Purpose: Detect supply chain issues before revenue impact
-- Prerequisite: sentiment_score populated via vader_sentiment.py
-- ============================================================

USE supply_chain_intelligence;

WITH monthly_seller_reviews AS (
    SELECT
        oi.seller_id,

        DATE_FORMAT(
            o.order_purchase_timestamp,
            '%Y-%m'
        ) AS review_month,

        COUNT(r.review_id) AS total_reviews,

        AVG(r.review_score) AS avg_review_score,

        AVG(r.sentiment_score) AS avg_sentiment,

        SUM(
            CASE
                WHEN r.review_score <= 2 THEN 1
                ELSE 0
            END
        ) AS bad_reviews,

        ROUND(
            100.0 *
            SUM(
                CASE
                    WHEN r.review_score <= 2 THEN 1
                    ELSE 0
                END
            )
            /
            NULLIF(COUNT(r.review_id),0),
            1
        ) AS bad_review_pct

    FROM olist_orders o
    JOIN olist_order_items oi
        ON o.order_id = oi.order_id
    JOIN olist_order_reviews r
        ON o.order_id = r.order_id

    WHERE
        r.review_score IS NOT NULL
        AND r.sentiment_score IS NOT NULL

    GROUP BY
        oi.seller_id,
        review_month
),

seller_perf AS (
    SELECT
        oi.seller_id,

        DATE_FORMAT(
            o.order_purchase_timestamp,
            '%Y-%m'
        ) AS perf_month,

        ROUND(
            100.0 *
            SUM(
                CASE
                    WHEN o.order_delivered_customer_date
                         <= o.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            )
            /
            NULLIF(COUNT(*),0),
            1
        ) AS on_time_pct

    FROM olist_orders o
    JOIN olist_order_items oi
        ON o.order_id = oi.order_id

    WHERE
        o.order_delivered_customer_date IS NOT NULL

    GROUP BY
        oi.seller_id,
        perf_month
)

SELECT
    msr.seller_id,

    msr.review_month,

    msr.total_reviews,

    ROUND(msr.avg_review_score,2)
        AS avg_star_rating,

    ROUND(msr.avg_sentiment,3)
        AS avg_sentiment_score,

    msr.bad_reviews,

    msr.bad_review_pct,

    COALESCE(sp.on_time_pct,0)
        AS on_time_delivery_pct,

    ROUND(
        100 - COALESCE(sp.on_time_pct,0),
        1
    ) AS delivery_failure_pct,

    CASE
        WHEN msr.avg_sentiment < -0.30
            THEN 'Strong Negative'

        WHEN msr.avg_sentiment < 0
            THEN 'Negative'

        WHEN msr.avg_sentiment < 0.20
            THEN 'Neutral'

        ELSE 'Positive'
    END AS sentiment_trend,

    ROUND(
        (
            msr.bad_review_pct * 0.50
        )
        +
        (
            (100 - COALESCE(sp.on_time_pct,100))
            * 0.30
        )
        +
        (
            ABS(msr.avg_sentiment) * 100
            * 0.20
        ),
        2
    ) AS supply_risk_score,

    CASE
        WHEN
            (
                msr.bad_review_pct * 0.50
                +
                (100 - COALESCE(sp.on_time_pct,100))
                * 0.30
                +
                ABS(msr.avg_sentiment) * 100
                * 0.20
            ) >= 40

        THEN 'CRITICAL'

        WHEN
            (
                msr.bad_review_pct * 0.50
                +
                (100 - COALESCE(sp.on_time_pct,100))
                * 0.30
                +
                ABS(msr.avg_sentiment) * 100
                * 0.20
            ) >= 25

        THEN 'WARNING'

        ELSE 'NORMAL'
    END AS early_warning_level,

    DENSE_RANK() OVER (
        ORDER BY
            (
                msr.bad_review_pct * 0.50
                +
                (100 - COALESCE(sp.on_time_pct,100))
                * 0.30
                +
                ABS(msr.avg_sentiment) * 100
                * 0.20
            ) DESC
    ) AS seller_risk_rank

FROM monthly_seller_reviews msr

LEFT JOIN seller_perf sp
    ON msr.seller_id = sp.seller_id
   AND msr.review_month = sp.perf_month

WHERE
    msr.total_reviews >= 5

ORDER BY
    supply_risk_score DESC,
    msr.bad_review_pct DESC,
    msr.review_month;