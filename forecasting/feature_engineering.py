"""
LSTM Feature Engineering
Pulls Q1 + Q8 SQL output from MySQL → builds time-series feature matrix.

Run:    python forecasting/feature_engineering.py
Output: data/processed/lstm_features.parquet
        data/processed/target_scaler.pkl
        data/processed/minmax_scaler.pkl
        data/processed/feature_dates.parquet
        data/processed/category_stats.json

Key design decisions:
- Cyclical encoding for month + day_of_week (sin/cos) — fixes mathematical
  discontinuity where month=12 and month=1 appear far apart to LSTM
- Brazil holidays added — Olist is Brazilian e-commerce, generic holidays wrong
- Calendar raw columns (month, day_of_week) dropped after cyclical encoding
- Calendar + binary features excluded from MinMaxScaler (already bounded)
- Dynamic top-N category fetch from MySQL — no hardcoded list
- Separate target scaler — required for correct inverse transform at inference
- Target pipeline: raw demand -> log1p -> MinMaxScaler. log1p compresses the
  right-skewed daily-demand distribution before scaling. Inverse must always
  go through to_raw_scale() (inverse_transform -> expm1) — never call
  target_scaler.inverse_transform() directly anywhere else.
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
import joblib
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("features")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DB_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER', 'root')}:"
    f"{os.getenv('MYSQL_PASSWORD', 'password')}@"
    f"{os.getenv('MYSQL_HOST', 'localhost')}:"
    f"{os.getenv('MYSQL_PORT', '3306')}/"
    f"{os.getenv('MYSQL_DATABASE', 'supply_chain_intelligence')}"
    "?charset=utf8mb4"
)

TOP_N           = 10
MIN_ACTIVE_DAYS = 60

# Only continuous unbounded features go through MinMaxScaler.
# Excluded (already bounded / binary / cyclical):
#   is_weekend (0/1), is_holiday (0/1),
#   month_sin/cos (-1 to 1), dow_sin/cos (-1 to 1),
#   days_to_holiday (bounded 0-30)
SCALE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7",  "rolling_std_30",
    "seasonality_index",
    "trend_strength", "acceleration",
]

FEATURE_COLS = [
    "units_sold",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",

    "rolling_mean_7",
    "rolling_mean_30",

    "rolling_std_7",
    "rolling_std_30",

    "seasonality_index",

    "month_sin",
    "month_cos",

    "dow_sin",
    "dow_cos",

    "is_weekend",
    "is_holiday",
    "days_to_holiday",

    "trend_strength",
    "acceleration",
]

# ── DB ────────────────────────────────────────────────────────────────────────

def get_engine():
    try:
        engine = create_engine(DB_URL, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("MySQL connection OK")
        return engine
    except Exception as e:
        log.error(f"MySQL connection failed: {e}")
        sys.exit(1)


# ── Brazil Holidays ───────────────────────────────────────────────────────────

def build_brazil_holiday_map(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Brazil-specific holidays using the `holidays` library.
    Generic holiday lists (US, global) are wrong for Olist — Brazilian
    e-commerce demand spikes around Carnival, Tiradentes, etc.

    days_to_holiday: days until next holiday (capped at 30).
    Captures pre-holiday demand surge — consumers buy before long weekends.
    """
    try:
        import holidays as hol
        br_holidays = hol.Brazil(years=list(range(
            date_range.min().year,
            date_range.max().year + 1
        )))
        holiday_dates = set(br_holidays.keys())
    except ImportError:
        log.warning("  `holidays` not installed — pip install holidays")
        log.warning("  Falling back to is_holiday=0, days_to_holiday=30 for all dates")
        return pd.DataFrame({
            "sale_date":      date_range,
            "is_holiday":     0,
            "days_to_holiday": 30,
        })

    records = []
    for d in date_range:
        date_only = d.date()
        is_hol    = int(date_only in holiday_dates)

        # Days to next holiday — capped at 30
        days_to = 30
        for ahead in range(1, 31):
            future = (d + pd.Timedelta(days=ahead)).date()
            if future in holiday_dates:
                days_to = ahead
                break

        records.append({
            "sale_date":       d,
            "is_holiday":      is_hol,
            "days_to_holiday": days_to,
        })

    holiday_df = pd.DataFrame(records)
    n_holidays = holiday_df["is_holiday"].sum()
    log.info(f"  Brazil holidays mapped: {n_holidays} holiday days "
             f"in {date_range.min().date()} → {date_range.max().date()}")
    return holiday_df


# ── Data Loading ──────────────────────────────────────────────────────────────

def fetch_top_categories(engine) -> list[str]:
    query = f"""
        SELECT
            p.product_category_name_english  AS category,
            COUNT(oi.order_item_id)          AS total_orders
        FROM olist_orders      o
        JOIN olist_order_items oi ON o.order_id    = oi.order_id
        JOIN olist_products    p  ON oi.product_id = p.product_id
        WHERE
            o.order_status = 'delivered'
            AND p.product_category_name_english IS NOT NULL
        GROUP BY category
        ORDER BY total_orders DESC
        LIMIT {TOP_N}
    """
    df = pd.read_sql(query, engine)
    log.info(f"Top {TOP_N} categories from MySQL:")
    for i, row in df.iterrows():
        log.info(f"  {i+1:>2}. {row['category']:<35} {row['total_orders']:>6,} orders")
    return df["category"].tolist()


def load_daily_demand(engine) -> pd.DataFrame:
    query = """
        SELECT
            DATE(o.order_purchase_timestamp)        AS sale_date,
            p.product_category_name_english         AS category,
            COUNT(oi.order_item_id)                 AS units_sold,
            SUM(oi.price)                           AS revenue
        FROM olist_orders      o
        JOIN olist_order_items oi ON o.order_id    = oi.order_id
        JOIN olist_products    p  ON oi.product_id = p.product_id
        WHERE
            o.order_status = 'delivered'
            AND p.product_category_name_english IS NOT NULL
        GROUP BY sale_date, category
        ORDER BY sale_date, category
    """
    df = pd.read_sql(query, engine, parse_dates=["sale_date"])
    if df.empty:
        log.error("No data returned — check MySQL tables")
        sys.exit(1)
    log.info(f"Daily demand: {len(df):,} rows | "
             f"{df['sale_date'].min().date()} → {df['sale_date'].max().date()}")
    return df


def load_seasonality_index(engine) -> pd.DataFrame:
    query = """
        WITH monthly_demand AS (
            SELECT
                MONTH(o.order_purchase_timestamp)        AS month_num,
                p.product_category_name_english          AS category,
                COUNT(oi.order_item_id)                  AS monthly_orders
            FROM olist_orders      o
            JOIN olist_order_items oi ON o.order_id    = oi.order_id
            JOIN olist_products    p  ON oi.product_id = p.product_id
            WHERE o.order_status = 'delivered'
            GROUP BY month_num, category
        ),
        annual_avg AS (
            SELECT category, AVG(monthly_orders) AS avg_orders
            FROM monthly_demand
            GROUP BY category
        )
        SELECT
            md.month_num,
            md.category,
            ROUND(100.0 * md.monthly_orders / NULLIF(aa.avg_orders, 0), 1) AS seasonality_index
        FROM monthly_demand md
        JOIN annual_avg     aa ON md.category = aa.category
    """
    df = pd.read_sql(query, engine)
    log.info(f"Seasonality index: {df['category'].nunique()} categories")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(daily_df: pd.DataFrame,
                      season_df: pd.DataFrame,
                      top_categories: list[str]) -> tuple[pd.DataFrame, dict]:

    df = daily_df[daily_df["category"].isin(top_categories)].copy()

    pivot = df.pivot_table(
        index="sale_date",
        columns="category",
        values="units_sold",
        aggfunc="sum",
    ).sort_index()

    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
    pivot    = pivot.reindex(full_idx, fill_value=0)
    pivot.index.name = "sale_date"

    # Build holiday map for full date range — done once, applied per category
    holiday_df = build_brazil_holiday_map(full_idx)

    feature_frames = []
    category_stats = {}

    for cat in top_categories:
        if cat not in pivot.columns:
            log.warning(f"  {cat}: not in pivot — skipping")
            continue

        series      = pivot[cat]
        total_days  = len(series)
        zero_days   = int((series == 0).sum())
        active_days = total_days - zero_days
        zero_pct    = zero_days / total_days * 100
        avg_daily   = float(series[series > 0].mean()) if active_days > 0 else 0.0

        category_stats[cat] = {
            "total_days":       total_days,
            "zero_days":        zero_days,
            "active_days":      active_days,
            "zero_pct":         round(zero_pct, 1),
            "avg_daily_demand": round(avg_daily, 2),
        }

        if active_days < MIN_ACTIVE_DAYS:
            log.warning(f"  {cat}: only {active_days} active days — skipping")
            continue

        if zero_pct > 40:
            log.warning(f"  {cat}: {zero_pct:.0f}% zero days — "
                        f"use SMAPE as primary metric")

        cat_df = pd.DataFrame({
            "sale_date":  pivot.index,
            "units_sold": series.values,
        })
        cat_df["category"] = cat

        # ── Lag features ─────────────────────────────────────────────────
        cat_df["lag_1"]  = cat_df["units_sold"].shift(1)
        cat_df["lag_7"]  = cat_df["units_sold"].shift(7)
        cat_df["lag_14"] = cat_df["units_sold"].shift(14)
        cat_df["lag_30"] = cat_df["units_sold"].shift(30)

        # ── Rolling stats ─────────────────────────────────────────────────
        cat_df["rolling_mean_7"]  = cat_df["units_sold"].rolling(7,  min_periods=1).mean()
        cat_df["rolling_mean_30"] = cat_df["units_sold"].rolling(30, min_periods=1).mean()
        cat_df["rolling_std_7"]   = cat_df["units_sold"].rolling(7,  min_periods=1).std().fillna(0)
        cat_df["rolling_std_30"]  = cat_df["units_sold"].rolling(30, min_periods=1).std().fillna(0)

        # ── Trend & acceleration ─────────────────────────────────────────
        # trend_strength: short vs. long-term momentum (7d avg minus 30d avg).
        #   Positive → demand running above its monthly baseline.
        # acceleration: lag_1 minus lag_7 — week-over-week point change.
        #   Note: this is a first difference, not a true second derivative —
        #   it flags short-term momentum shifts the rolling means smooth away.
        # Both are unbounded differences (can be negative), so they're added
        # to SCALE_COLS below to go through MinMaxScaler like the other
        # continuous features.
        cat_df["trend_strength"] = cat_df["rolling_mean_7"] - cat_df["rolling_mean_30"]
        cat_df["acceleration"]   = cat_df["lag_1"] - cat_df["lag_7"]

        # ── Cyclical encoding ─────────────────────────────────────────────
        # Replaces raw month + day_of_week.
        # Problem with raw encoding: LSTM sees month=12 and month=1 as distance=11,
        # but they are actually adjacent (December → January).
        # Sin/cos projects onto a circle: distance(Dec, Jan) = distance(Jan, Feb) ✓
        month = cat_df["sale_date"].dt.month
        dow   = cat_df["sale_date"].dt.dayofweek

        cat_df["month_sin"] = np.sin(2 * np.pi * month / 12)
        cat_df["month_cos"] = np.cos(2 * np.pi * month / 12)
        cat_df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
        cat_df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)

        # Binary calendar — not normalized
        cat_df["is_weekend"] = (dow >= 5).astype(int)

        # ── Seasonality index (from Q8) ───────────────────────────────────
        season_cat  = season_df[season_df["category"] == cat][["month_num", "seasonality_index"]]
        season_map  = dict(zip(season_cat["month_num"], season_cat["seasonality_index"]))
        cat_df["seasonality_index"] = month.map(season_map).fillna(100.0)

        # ── Brazil holidays ───────────────────────────────────────────────
        cat_df = cat_df.merge(
            holiday_df[["sale_date", "is_holiday", "days_to_holiday"]],
            on="sale_date",
            how="left",
        )
        cat_df["is_holiday"]      = cat_df["is_holiday"].fillna(0).astype(int)
        cat_df["days_to_holiday"] = cat_df["days_to_holiday"].fillna(30).astype(int)

        feature_frames.append(cat_df)

    if not feature_frames:
        log.error("No categories passed quality checks")
        sys.exit(1)

    combined = pd.concat(feature_frames, ignore_index=True)
    before   = len(combined)
    combined = combined.dropna(subset=FEATURE_COLS)
    dropped  = before - len(combined)

    log.info(f"Feature matrix: {combined.shape} | "
             f"Dropped {dropped:,} NaN rows from lag creation")
    return combined, category_stats


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_and_save(df: pd.DataFrame) -> pd.DataFrame:
    # Target pipeline:
    #   raw demand
    #   -> log1p
    #   -> MinMaxScaler
    #
    # log1p compresses the right-skewed daily-demand distribution (many small
    # days, occasional large spikes) before scaling. This MUST be mirrored at
    # inference time by to_raw_scale(): inverse_transform() -> np.expm1().
    # Do not call target_scaler.inverse_transform() directly anywhere else in
    # the project — always route predictions back to raw units through
    # to_raw_scale() so there is exactly one place doing inverse_transform -> expm1.
    df["units_sold"] = np.log1p(df["units_sold"])

    target_scaler = MinMaxScaler()
    df[["units_sold"]] = target_scaler.fit_transform(df[["units_sold"]])
    joblib.dump(target_scaler, PROCESSED_DIR / "target_scaler.pkl")

    # Feature scaler — continuous unbounded features only
    # Cyclical (sin/cos) already in [-1,1] — no scaling needed
    # Binary (is_weekend, is_holiday) — scaling pointless on 0/1
    # days_to_holiday — bounded [0,30] — acceptable to scale but low impact
    feature_scaler = MinMaxScaler()
    df[SCALE_COLS] = feature_scaler.fit_transform(df[SCALE_COLS])
    joblib.dump(feature_scaler, PROCESSED_DIR / "minmax_scaler.pkl")

    # Save dates separately — kept out of feature parquet to avoid dtype mixing
    dates_df = df[["sale_date", "category"]].copy()
    dates_df.to_parquet(PROCESSED_DIR / "feature_dates.parquet", index=False)

    # Feature parquet — only FEATURE_COLS + category
    out_df = df[FEATURE_COLS + ["category"]].copy()
    out_df.to_parquet(PROCESSED_DIR / "lstm_features.parquet", index=False)

    log.info(f"target_scaler.pkl  saved")
    log.info(f"minmax_scaler.pkl  saved")
    log.info(f"feature_dates.parquet saved  ({len(dates_df):,} rows)")
    log.info(f"lstm_features.parquet saved  ({len(out_df):,} rows, {len(FEATURE_COLS)} features)")
    return out_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("LSTM FEATURE ENGINEERING")
    log.info(f"Features: {len(FEATURE_COLS)} | Top categories: {TOP_N}")
    log.info("=" * 55)

    engine         = get_engine()
    top_categories = fetch_top_categories(engine)
    daily_df       = load_daily_demand(engine)
    season_df      = load_seasonality_index(engine)

    features, category_stats = engineer_features(daily_df, season_df, top_categories)
    normalize_and_save(features)

    with open(PROCESSED_DIR / "category_stats.json", "w") as f:
        json.dump(category_stats, f, indent=2)

    # Final report
    log.info("\nCategory data quality:")
    log.info(f"  {'Category':<35} {'Active':>6} {'Zero%':>6} {'Avg/day':>8}")
    log.info("  " + "-" * 60)
    for cat, s in category_stats.items():
        flag = " ⚠" if s["zero_pct"] > 40 else ""
        log.info(f"  {cat:<35} {s['active_days']:>6} "
                 f"{s['zero_pct']:>5.1f}% {s['avg_daily_demand']:>8.1f}{flag}")

    processed = features["category"].nunique()
    log.info(f"\n{processed}/{len(top_categories)} categories processed")
    log.info(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    log.info("Next → python forecasting/baseline_models.py")


if __name__ == "__main__":
    main()