-- ============================================================
-- Q7 — Revenue at Risk from Delayed Deliveries
-- Techniques: Conditional SUM, DATEDIFF, CASE, Risk Scoring
-- Purpose: Quantify financial impact of supplier delays
-- ============================================================

USE supply_chain_intelligence;

SELECT
    oi.seller_id,
    s.seller_state,

    COUNT(DISTINCT o.order_id) AS total_orders,

    SUM(
        CASE
            WHEN o.order_delivered_customer_date >
                 o.order_estimated_delivery_date
            THEN 1
            ELSE 0
        END
    ) AS delayed_orders,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN o.order_delivered_customer_date >
                     o.order_estimated_delivery_date
                THEN 1
                ELSE 0
            END
        ) /
        COUNT(DISTINCT o.order_id),
        1
    ) AS delay_rate_pct,

    -- Total revenue handled by seller
    ROUND(
        SUM(oi.price + oi.freight_value),
        2
    ) AS total_revenue,

    -- Revenue from delayed orders
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
    ) AS revenue_at_risk,

    -- Revenue risk as percentage
    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN o.order_delivered_customer_date >
                     o.order_estimated_delivery_date
                THEN oi.price + oi.freight_value
                ELSE 0
            END
        )
        /
        SUM(oi.price + oi.freight_value),
        2
    ) AS revenue_risk_pct,

    -- Average delay
    ROUND(
        AVG(
            CASE
                WHEN o.order_delivered_customer_date >
                     o.order_estimated_delivery_date
                THEN DATEDIFF(
                    o.order_delivered_customer_date,
                    o.order_estimated_delivery_date
                )
            END
        ),
        1
    ) AS avg_days_late,

    -- Maximum delay
    MAX(
        CASE
            WHEN o.order_delivered_customer_date >
                 o.order_estimated_delivery_date
            THEN DATEDIFF(
                o.order_delivered_customer_date,
                o.order_estimated_delivery_date
            )
            ELSE 0
        END
    ) AS max_days_late,

    -- Delay severity category
    CASE
        WHEN AVG(
            CASE
                WHEN o.order_delivered_customer_date >
                     o.order_estimated_delivery_date
                THEN DATEDIFF(
                    o.order_delivered_customer_date,
                    o.order_estimated_delivery_date
                )
            END
        ) > 10
            THEN 'SEVERE'

        WHEN AVG(
            CASE
                WHEN o.order_delivered_customer_date >
                     o.order_estimated_delivery_date
                THEN DATEDIFF(
                    o.order_delivered_customer_date,
                    o.order_estimated_delivery_date
                )
            END
        ) > 5
            THEN 'MODERATE'

        ELSE 'MINOR'
    END AS delay_severity,

    -- Composite supplier risk score
    ROUND(
        (
            (
                100.0 *
                SUM(
                    CASE
                        WHEN o.order_delivered_customer_date >
                             o.order_estimated_delivery_date
                        THEN 1
                        ELSE 0
                    END
                )
                /
                COUNT(DISTINCT o.order_id)
            ) * 0.6
        )
        +
        (
            AVG(
                CASE
                    WHEN o.order_delivered_customer_date >
                         o.order_estimated_delivery_date
                    THEN DATEDIFF(
                        o.order_delivered_customer_date,
                        o.order_estimated_delivery_date
                    )
                END
            ) * 0.4
        ),
        2
    ) AS supplier_risk_score

FROM olist_orders o
JOIN olist_order_items oi
    ON o.order_id = oi.order_id
JOIN olist_sellers s
    ON oi.seller_id = s.seller_id

WHERE
    o.order_delivered_customer_date IS NOT NULL
    AND o.order_estimated_delivery_date IS NOT NULL

GROUP BY
    oi.seller_id,
    s.seller_state

HAVING
    total_orders >= 10
    AND delayed_orders > 0

ORDER BY
    supplier_risk_score DESC,
    revenue_at_risk DESC

LIMIT 50;