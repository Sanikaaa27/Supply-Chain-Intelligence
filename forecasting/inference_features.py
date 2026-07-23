"""
Inference-Time Feature Engineering — for FastAPI /forecast endpoint.

This is the SERVING counterpart to forecasting/feature_engineering.py.
That script is a BATCH job — pulls ALL history, ALL categories, fits new
scalers, writes parquet files. It runs ONCE, offline, when you rerun the
training pipeline.

This file does the OPPOSITE: given a single category + a single as-of-date,
it pulls only the recent history needed (lookback window), recomputes the
exact same 19 features for the LATEST row only, and transforms them using
the ALREADY-FITTED scalers saved to disk. It NEVER calls fit() or
fit_transform() on anything — only .transform(). If it did fit a new
scaler here, every prediction would be scaled differently than the model
was trained on, and forecasts would be silently wrong (no crash, just bad
numbers).

Usage in FastAPI:

    from forecasting.inference_features import build_inference_row

    row = build_inference_row("telephony")
    # row is a single-row DataFrame, ready to feed straight into
    # gbm_model.predict_horizon(model, row) for every horizon model.

Run standalone for a sanity check:
    python forecasting/inference_features.py telephony
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("inference_features")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "data" / "models"

DB_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER', 'root')}:"
    f"{os.getenv('MYSQL_PASSWORD', 'password')}@"
    f"{os.getenv('MYSQL_HOST', 'localhost')}:"
    f"{os.getenv('MYSQL_PORT', '3306')}/"
    f"{os.getenv('MYSQL_DATABASE', 'supply_chain_intelligence')}"
    "?charset=utf8mb4"
)

# Must match feature_engineering.py EXACTLY — same order, same names.
FEATURE_COLS = [
    "units_sold",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30",
    "seasonality_index",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "is_weekend", "is_holiday", "days_to_holiday",
    "trend_strength", "acceleration",
]

# Must match feature_engineering.py's SCALE_COLS exactly — same columns
# go through the feature MinMaxScaler. Everything else (cyclical, binary,
# units_sold which gets its OWN target scaler) is excluded.
SCALE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30",
    "seasonality_index",
    "trend_strength", "acceleration",
]

# Need at least 30 days back for lag_30/rolling_mean_30, plus a safety
# buffer for holiday/seasonality lookups and to avoid edge effects from
# rolling windows on too-short a slice. 60 is a deliberate safety margin,
# not a magic number copied from training (training used full history).
LOOKBACK_DAYS = 60
INFERENCE_CACHE_PATH = PROCESSED_DIR / 'inference_cache.csv'

def load_from_cache(category: str) -> pd.DataFrame:
    if not INFERENCE_CACHE_PATH.exists():
        raise InferenceFeatureError('inference_cache.csv not found in data/processed/')
    df = pd.read_csv(INFERENCE_CACHE_PATH)
    if category not in df['category'].values:
        available = df['category'].tolist()
        raise InferenceFeatureError(f'Category {category!r} not in cache. Available: {available}')
    row = df[df['category'] == category].iloc[[-1]].copy()
    known_categories = load_known_categories()
    cat_dtype = pd.CategoricalDtype(categories=known_categories)
    row['category'] = row['category'].astype(cat_dtype)
    row['anchor'] = row['units_sold'].values
    return row[['category', 'anchor'] + FEATURE_COLS]



class InferenceFeatureError(RuntimeError):
    """Raised when there isn't enough recent history to build a valid
    feature row (e.g. brand-new category, gap in data). The API layer
    should catch this and return a 422/503 — never silently predict on
    incomplete/garbage features."""


# ── DB ────────────────────────────────────────────────────────────────────

def get_engine():
    try:
        engine = create_engine(DB_URL, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        raise InferenceFeatureError(f"MySQL connection failed: {e}") from e


def load_recent_daily_demand(engine, category: str, as_of_date: date,
                              lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Pulls only this category's last `lookback_days` of daily demand,
    up to and including as_of_date. Mirrors load_daily_demand() in
    feature_engineering.py but scoped to one category and a bounded window
    instead of the full table — this is the part that makes serving fast."""
    start_date = as_of_date - timedelta(days=lookback_days)

    query = """
        SELECT
            DATE(o.order_purchase_timestamp)        AS sale_date,
            COUNT(oi.order_item_id)                 AS units_sold
        FROM olist_orders      o
        JOIN olist_order_items oi ON o.order_id    = oi.order_id
        JOIN olist_products    p  ON oi.product_id = p.product_id
        WHERE
            o.order_status = 'delivered'
            AND p.product_category_name_english = :category
            AND DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :as_of_date
        GROUP BY sale_date
        ORDER BY sale_date
    """
    df = pd.read_sql(
        text(query), engine,
        params={"category": category, "start_date": start_date, "as_of_date": as_of_date},
        parse_dates=["sale_date"],
    )

    full_idx = pd.date_range(start_date, as_of_date, freq="D")
    series = df.set_index("sale_date")["units_sold"].reindex(full_idx, fill_value=0)
    series.index.name = "sale_date"
    return series.reset_index().rename(columns={"sale_date": "sale_date"})


def load_seasonality_index_for_category(engine, category: str) -> dict:
    """Same Q8 logic as feature_engineering.py's load_seasonality_index(),
    scoped to one category. month_num -> seasonality_index."""
    query = """
        WITH monthly_demand AS (
            SELECT
                MONTH(o.order_purchase_timestamp)        AS month_num,
                COUNT(oi.order_item_id)                  AS monthly_orders
            FROM olist_orders      o
            JOIN olist_order_items oi ON o.order_id    = oi.order_id
            JOIN olist_products    p  ON oi.product_id = p.product_id
            WHERE o.order_status = 'delivered'
              AND p.product_category_name_english = :category
            GROUP BY month_num
        ),
        annual_avg AS (
            SELECT AVG(monthly_orders) AS avg_orders FROM monthly_demand
        )
        SELECT
            md.month_num,
            ROUND(100.0 * md.monthly_orders / NULLIF(aa.avg_orders, 0), 1) AS seasonality_index
        FROM monthly_demand md
        JOIN annual_avg aa
    """
    df = pd.read_sql(text(query), engine, params={"category": category})
    return dict(zip(df["month_num"], df["seasonality_index"]))


def get_brazil_holiday_info(d: pd.Timestamp) -> tuple[int, int]:
    """is_holiday, days_to_holiday for a single date — same logic/library
    as build_brazil_holiday_map() in feature_engineering.py, just evaluated
    for one date instead of a full range (no need to build the whole map
    for a single inference call)."""
    try:
        import holidays as hol
        br_holidays = hol.Brazil(years=[d.year, d.year + 1])
    except ImportError:
        log.warning("`holidays` not installed — is_holiday=0, days_to_holiday=30")
        return 0, 30

    holiday_dates = set(br_holidays.keys())
    is_hol = int(d.date() in holiday_dates)

    days_to = 30
    for ahead in range(1, 31):
        if (d + pd.Timedelta(days=ahead)).date() in holiday_dates:
            days_to = ahead
            break
    return is_hol, days_to


# ── Feature construction ─────────────────────────────────────────────────

def compute_latest_features(series_df: pd.DataFrame, category: str,
                             season_map: dict) -> pd.DataFrame:
    """Recomputes the exact same 19 features as feature_engineering.py's
    engineer_features(), but only keeps the LAST row (the as-of-date row),
    since that's the only one a forecast is ever made from."""
    df = series_df.copy()
    df["units_sold"] = df["units_sold"].astype(float)

    df["lag_1"]  = df["units_sold"].shift(1)
    df["lag_7"]  = df["units_sold"].shift(7)
    df["lag_14"] = df["units_sold"].shift(14)
    df["lag_30"] = df["units_sold"].shift(30)

    df["rolling_mean_7"]  = df["units_sold"].rolling(7,  min_periods=1).mean()
    df["rolling_mean_30"] = df["units_sold"].rolling(30, min_periods=1).mean()
    df["rolling_std_7"]   = df["units_sold"].rolling(7,  min_periods=1).std().fillna(0)
    df["rolling_std_30"]  = df["units_sold"].rolling(30, min_periods=1).std().fillna(0)

    df["trend_strength"] = df["rolling_mean_7"] - df["rolling_mean_30"]
    df["acceleration"]   = df["lag_1"] - df["lag_7"]

    month = df["sale_date"].dt.month
    dow   = df["sale_date"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
    df["is_weekend"] = (dow >= 5).astype(int)

    df["seasonality_index"] = month.map(season_map).fillna(100.0)

    last_date = df["sale_date"].iloc[-1]
    is_hol, days_to_hol = get_brazil_holiday_info(last_date)
    df["is_holiday"] = 0
    df["days_to_holiday"] = 30
    df.iloc[-1, df.columns.get_loc("is_holiday")] = is_hol
    df.iloc[-1, df.columns.get_loc("days_to_holiday")] = days_to_hol

    latest = df.iloc[[-1]].copy()

    missing_lags = latest[["lag_1", "lag_7", "lag_14", "lag_30"]].isna().any(axis=1).iloc[0]
    if missing_lags:
        raise InferenceFeatureError(
            f"Not enough history for '{category}' to compute lag_30 — "
            f"need >= 30 days of data before as-of-date, got {len(df)} rows."
        )

    latest["category"] = category
    return latest[["sale_date", "category"] + FEATURE_COLS]


# ── Scaling (transform ONLY — never fit) ─────────────────────────────────

def load_scalers():
    target_path  = PROCESSED_DIR / "target_scaler.pkl"
    feature_path = PROCESSED_DIR / "minmax_scaler.pkl"
    if not target_path.exists() or not feature_path.exists():
        raise InferenceFeatureError(
            "Scalers not found in data/processed/. Run feature_engineering.py "
            "(training pipeline) at least once before serving."
        )
    return joblib.load(target_path), joblib.load(feature_path)


def load_known_categories() -> list[str]:
    """Pulls the training-time category list + order from model_selection.json
    so the 'category' column can be cast to the EXACT same pandas
    CategoricalDtype the GBM models were trained with. LightGBM's native
    categorical handling is order/identity sensitive — get this wrong and
    predictions are silently corrupted, not an error."""
    path = PROCESSED_DIR / "model_selection.json"
    if not path.exists():
        raise InferenceFeatureError(
            f"{path} not found — run model_comparison.py first."
        )
    with open(path) as f:
        data = json.load(f)
    return sorted(data["per_category"].keys())


def scale_row(row: pd.DataFrame, target_scaler, feature_scaler,
              known_categories: list[str]) -> pd.DataFrame:
    """Applies the SAME log1p -> MinMaxScaler pipeline as training, using
    .transform() (never .fit/.fit_transform). Mirrors normalize_and_save()
    in feature_engineering.py but for one row, with pre-fitted scalers."""
    out = row.copy()

    # Target/units_sold pipeline: log1p -> target_scaler.transform
    # (must match normalize_and_save(): same order, same scaler object).
    out["units_sold"] = np.log1p(out["units_sold"])
    out[["units_sold"]] = target_scaler.transform(out[["units_sold"]])

    # Feature scaler — only the continuous unbounded columns, exactly like
    # training. Cyclical/binary/bounded columns are left untouched.
    out[SCALE_COLS] = feature_scaler.transform(out[SCALE_COLS])

    # "anchor" — same convention as gbm_model.py's build_feature_matrix():
    # the (scaled) units_sold value the forecast is anchored from.
    out["anchor"] = out["units_sold"].values

    # Cast category to the EXACT training-time CategoricalDtype.
    if row["category"].iloc[0] not in known_categories:
        log.warning(
            f"Category '{row['category'].iloc[0]}' was not in training data "
            f"({len(known_categories)} known categories) — model_selector's "
            f"global-default fallback should be used for model CHOICE, but "
            f"the category dtype below will still include it as a new level, "
            f"which most GBM categorical splits will simply never trigger on."
        )
    cat_dtype = pd.CategoricalDtype(categories=known_categories)
    out["category"] = out["category"].astype(cat_dtype)

    return out


# ── Public entry point ────────────────────────────────────────────────────

# Olist's last ~7-10 days of order data is incomplete (orders still being
# processed/recorded when the Kaggle snapshot was taken) — confirmed
# empirically: daily order counts fall off a cliff in the final days
# (e.g. ~250/day around day -10 down to ~11/day on the absolute last day).
# Using the absolute max date as as_of_date silently anchors every forecast
# on artificially low/zero demand. This buffer pulls back into the stable
# zone instead of guessing — DATASET_TAIL_BUFFER_DAYS should be re-checked
# if the underlying data is ever refreshed/re-exported.
DATASET_TAIL_BUFFER_DAYS = 10


def get_dataset_latest_date(engine, buffer_days: int = DATASET_TAIL_BUFFER_DAYS) -> date:
    """Olist data is historical (2016-2018) — it does NOT continue up to
    real-world 'today'. Using date.today() as the default as_of_date would
    silently query a date range with zero matching rows, and every feature
    would come back 0.0 (no crash — that's exactly what happened on the
    first test run). The correct 'now' for this dataset is its own most
    recent order date, MINUS a buffer to skip the incomplete tail (see
    DATASET_TAIL_BUFFER_DAYS above) — without the buffer, anchor/units_sold
    comes back artificially near-zero for most categories (also confirmed
    empirically: 3 of 4 test categories showed anchor=0.0 before this fix)."""
    query = "SELECT MAX(DATE(order_purchase_timestamp)) AS max_date FROM olist_orders WHERE order_status = 'delivered'"
    result = pd.read_sql(text(query), engine)
    max_date = result["max_date"].iloc[0]
    if max_date is None:
        raise InferenceFeatureError("Could not determine dataset's latest date — olist_orders empty?")
    max_date = max_date if isinstance(max_date, date) else pd.Timestamp(max_date).date()
    return max_date - timedelta(days=buffer_days)

def build_inference_row(category: str, as_of_date: date | None = None,
                         engine=None) -> pd.DataFrame:
    """The one function the FastAPI endpoint should call.

    Returns a single-row DataFrame with columns:
        FEATURE_COLS (19, scaled) + ["category" (correct dtype), "anchor"]
    Ready to pass straight into gbm_model.predict_horizon(model, row) for
    every horizon-day model, or into the LSTM's prediction path.

    as_of_date: if None, defaults to the DATASET's latest available date
    (NOT date.today() — this is historical 2016-2018 data, "today" in the
    real world has zero matching rows and would silently return all-zero
    features instead of erroring).
    """
    if engine is None:
        try:
            engine = get_engine()
        except InferenceFeatureError:
            log.warning("MySQL unavailable — falling back to inference_cache.csv")
            return load_from_cache(category)

    as_of_date = as_of_date or get_dataset_latest_date(engine)

    try:
        series_df = load_recent_daily_demand(engine, category, as_of_date)
        season_map = load_seasonality_index_for_category(engine, category)
        latest = compute_latest_features(series_df, category, season_map)

        target_scaler, feature_scaler = load_scalers()
        known_categories = load_known_categories()
        scaled_row = scale_row(latest, target_scaler, feature_scaler, known_categories)

        return scaled_row.drop(columns=["sale_date"])
    finally:
        engine.dispose()


# ── LSTM sequence building (different shape requirement than GBM) ────────
#
# GBM needs ONE row of features. The LSTM needs a SEQUENCE: lookback=30
# consecutive days, each with all 19 features, fed as (1, 30, 19) — see
# lstm_model.py's build_sequences(). This is NOT the same as calling
# build_inference_row() 30 times; the lag/rolling features for EACH day in
# the sequence must be computed from ITS OWN preceding history, exactly as
# they were during training (lag_30 on day t needs days t-30..t-1, not the
# single as-of-date's lookback).

LSTM_LOOKBACK = 30  # must match lstm_global_smoothed_metadata.json "lookback"


def compute_full_feature_series(series_df: pd.DataFrame, category: str,
                                 season_map: dict) -> pd.DataFrame:
    """Same feature logic as compute_latest_features(), but returns EVERY
    valid row (post lag-30 warmup), not just the last one. Used to build
    an LSTM sequence instead of a single GBM row."""
    df = series_df.copy()
    df["units_sold"] = df["units_sold"].astype(float)

    df["lag_1"]  = df["units_sold"].shift(1)
    df["lag_7"]  = df["units_sold"].shift(7)
    df["lag_14"] = df["units_sold"].shift(14)
    df["lag_30"] = df["units_sold"].shift(30)

    df["rolling_mean_7"]  = df["units_sold"].rolling(7,  min_periods=1).mean()
    df["rolling_mean_30"] = df["units_sold"].rolling(30, min_periods=1).mean()
    df["rolling_std_7"]   = df["units_sold"].rolling(7,  min_periods=1).std().fillna(0)
    df["rolling_std_30"]  = df["units_sold"].rolling(30, min_periods=1).std().fillna(0)

    df["trend_strength"] = df["rolling_mean_7"] - df["rolling_mean_30"]
    df["acceleration"]   = df["lag_1"] - df["lag_7"]

    month = df["sale_date"].dt.month
    dow   = df["sale_date"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
    df["is_weekend"] = (dow >= 5).astype(int)

    df["seasonality_index"] = month.map(season_map).fillna(100.0)

    # Holiday info per-day (small lookup, but it's a fixed ~95-day window —
    # cheap enough to compute per row rather than building the full-range
    # map like training did).
    is_hol_list, days_to_list = [], []
    for d in df["sale_date"]:
        ih, dh = get_brazil_holiday_info(d)
        is_hol_list.append(ih)
        days_to_list.append(dh)
    df["is_holiday"]      = is_hol_list
    df["days_to_holiday"] = days_to_list

    df["category"] = category
    valid = df.dropna(subset=FEATURE_COLS)
    return valid[["sale_date", "category"] + FEATURE_COLS]


def load_lstm_category_id_map() -> dict:
    """Loads the EXACT cat_to_id mapping the LSTM embedding layer was
    trained with — must use this file, not re-derive it, since
    sorted(categories) happens to match here but re-deriving it independently
    in two places is exactly the kind of duplicated assumption that breaks
    silently if either side ever changes."""
    path = MODELS_DIR / "lstm_global_smoothed_categories.json"
    if not path.exists():
        raise InferenceFeatureError(f"{path} not found — train the LSTM model first.")
    with open(path) as f:
        return json.load(f)


def build_inference_sequence(category: str, as_of_date: date | None = None,
                              lookback: int = LSTM_LOOKBACK,
                              engine=None) -> dict:
    """LSTM equivalent of build_inference_row(). Returns a dict with:
        "X"        : np.ndarray, shape (1, lookback, 19) — scaled features
        "category_id" : np.ndarray, shape (1,) int32 — for the embedding input
        "anchor"   : np.ndarray, shape (1,) — scaled units_sold of the last
                     day in the sequence (add this to the model's delta
                     output, in scaled space, before inverse-transforming —
                     same convention as lstm_model.py's build_sequences()).

    Ready to call: model.predict([X, category_id]) -> delta (1, 7)
                    forecast_scaled = delta + anchor.reshape(-1, 1)
                    forecast_raw = to_raw_scale(forecast_scaled, target_scaler, clip_negative=True)
    """
    owns_engine = engine is None
    engine = engine or get_engine()
    as_of_date = as_of_date or get_dataset_latest_date(engine)

    try:
        # Fetch enough raw history: lookback days of SEQUENCE + 30 more for
        # the lag_30/rolling_30 warmup of the EARLIEST day in that sequence.
        fetch_window = lookback + 35
        series_df = load_recent_daily_demand(engine, category, as_of_date, lookback_days=fetch_window)
        season_map = load_seasonality_index_for_category(engine, category)
        full_features = compute_full_feature_series(series_df, category, season_map)

        if len(full_features) < lookback:
            raise InferenceFeatureError(
                f"Not enough valid feature rows for '{category}' to build an "
                f"LSTM sequence — need {lookback}, got {len(full_features)}."
            )

        sequence_df = full_features.tail(lookback).reset_index(drop=True)

        target_scaler, feature_scaler = load_scalers()

        scaled = sequence_df.copy()
        scaled["units_sold"] = np.log1p(scaled["units_sold"])
        scaled[["units_sold"]] = target_scaler.transform(scaled[["units_sold"]])
        scaled[SCALE_COLS] = feature_scaler.transform(scaled[SCALE_COLS])

        X = scaled[FEATURE_COLS].values.astype(np.float32).reshape(1, lookback, len(FEATURE_COLS))
        anchor = np.array([scaled["units_sold"].iloc[-1]], dtype=np.float32)

        cat_to_id = load_lstm_category_id_map()
        if category not in cat_to_id:
            raise InferenceFeatureError(
                f"'{category}' has no entry in lstm_global_smoothed_categories.json "
                f"— the LSTM embedding layer has no learned vector for it, predicting "
                f"would use an arbitrary/wrong embedding."
            )
        category_id = np.array([cat_to_id[category]], dtype=np.int32)

        return {"X": X, "category_id": category_id, "anchor": anchor}
    finally:
        if owns_engine:
            engine.dispose()


# ── Manual check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python forecasting/inference_features.py <category> [--lstm]")
        sys.exit(1)

    cat = sys.argv[1]
    if "--lstm" in sys.argv:
        seq = build_inference_sequence(cat)
        print(f"\nLSTM sequence for '{cat}':")
        print(f"  X shape: {seq['X'].shape}  (expect (1, 30, 19))")
        print(f"  category_id: {seq['category_id']}")
        print(f"  anchor (scaled): {seq['anchor']}")
    else:
        row = build_inference_row(cat)
        print(f"\nInference row for '{cat}':\n")
        print(row.T)
