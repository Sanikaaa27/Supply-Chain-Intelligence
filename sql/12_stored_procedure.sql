-- ============================================================
-- Q12 — Stored Procedure: Automated Monthly Supply Report
-- Techniques: CREATE PROCEDURE, parameters, multiple result sets
-- Usage: CALL monthly_supply_report('2018-10-01');
-- ============================================================

USE supply_chain_intelligence;

DROP PROCEDURE IF EXISTS monthly_supply_report;

DELIMITER $$

CREATE PROCEDURE monthly_supply_report(IN report_month DATE)
BEGIN
    -- Derive month bounds from input date
    DECLARE v_month_start DATE DEFAULT DATE_FORMAT(report_month, '%Y-%m-01');
    DECLARE v_month_end   DATE DEFAULT LAST_DAY(report_month);
    DECLARE v_month_label VARCHAR(7) DEFAULT DATE_FORMAT(report_month, '%Y-%m');

    -- ── Result Set 1: Report Header ────────────────────────────────────────
    SELECT
        'MONTHLY SUPPLY CHAIN REPORT'               AS report_title,
        v_month_label                               AS report_period,
        NOW()                                       AS generated_at;

    -- ── Result Set 2: Top 5 Delayed Sellers ───────────────────────────────
    SELECT
        'TOP 5 DELAYED SELLERS'                     AS section,
        oi.seller_id,
        COUNT(*)                                    AS total_orders,
        SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
            THEN 1 ELSE 0 END)                      AS delayed_orders,
        ROUND(
            100.0 * SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                THEN 1 ELSE 0 END) / COUNT(*),
            1
        )                                           AS delay_pct,
        ROUND(SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
            THEN oi.price ELSE 0 END), 2)           AS revenue_at_risk
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE
        o.order_purchase_timestamp BETWEEN v_month_start AND v_month_end
        AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY oi.seller_id
    HAVING total_orders >= 5
    ORDER BY delay_pct DESC
    LIMIT 5;

    -- ── Result Set 3: Stockout Category Alerts ────────────────────────────
    SELECT
        'STOCKOUT ALERTS'                           AS section,
        p.product_category_name_english             AS category,
        COUNT(DISTINCT oi.product_id)               AS products_at_risk,
        SUM(oi.price)                               AS total_category_revenue,
        MIN(o.order_purchase_timestamp)             AS first_signal_date
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id    = oi.order_id
    JOIN olist_products    p  ON oi.product_id = p.product_id
    WHERE
        o.order_purchase_timestamp BETWEEN v_month_start AND v_month_end
        AND o.order_status IN ('canceled', 'unavailable')
    GROUP BY category
    ORDER BY products_at_risk DESC
    LIMIT 10;

    -- ── Result Set 4: Revenue at Risk Summary ─────────────────────────────
    SELECT
        'REVENUE AT RISK SUMMARY'                   AS section,
        COUNT(DISTINCT o.order_id)                  AS total_orders_this_month,
        SUM(oi.price + oi.freight_value)            AS total_revenue,
        SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
            THEN oi.price + oi.freight_value ELSE 0 END) AS revenue_at_risk,
        ROUND(
            100.0 * SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT o.order_id), 0),
            1
        )                                           AS overall_delay_pct
    FROM olist_orders      o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE
        o.order_purchase_timestamp BETWEEN v_month_start AND v_month_end
        AND o.order_delivered_customer_date IS NOT NULL;

    -- ── Result Set 5: Reorder Alerts (products below reorder point) ───────
    SELECT
        'REORDER ALERTS'                            AS section,
        p.product_category_name_english             AS category,
        oi.product_id,
        COUNT(oi.order_item_id)                     AS units_sold_this_month,
        -- Products with high velocity but low recent orders signal stockout risk
        CASE
            WHEN COUNT(oi.order_item_id) < 5  THEN 'CRITICAL — Reorder now'
            WHEN COUNT(oi.order_item_id) < 15 THEN 'LOW STOCK — Review needed'
            ELSE 'ADEQUATE'
        END                                         AS reorder_status
    FROM olist_order_items oi
    JOIN olist_orders      o  ON oi.order_id    = o.order_id
    JOIN olist_products    p  ON oi.product_id  = p.product_id
    WHERE
        o.order_purchase_timestamp BETWEEN v_month_start AND v_month_end
    GROUP BY p.product_category_name_english, oi.product_id
    HAVING reorder_status != 'ADEQUATE'
    ORDER BY units_sold_this_month ASC
    LIMIT 20;

END$$

DELIMITER ;

-- ── Test the stored procedure ─────────────────────────────────────────────
-- CALL monthly_supply_report('2018-10-01');
-- CALL monthly_supply_report('2018-08-01');
