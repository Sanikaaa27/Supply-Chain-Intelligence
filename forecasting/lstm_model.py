"""LSTM Demand Forecasting — Production Pipeline (cleaned)

Single experiment: 7-day smoothed demand (rolling mean target, log-transformed).
Global model: one shared LSTM stack + a learned category embedding,
instead of training 10 separate per-category models.

Run:
    python lstm_model.py                  # full walk-forward CV + final training
    python lstm_model.py --skip-cv        # final training only (~4x faster)
    python lstm_model.py --tune-lookback --tune-hp
    python lstm_model.py --use-layernorm  # LayerNorm instead of BatchNorm

Architecture:
    LSTM(32) -> BatchNorm -> Dropout(0.4) -> LSTM(16) -> Dropout(0.4)
    -> concat with category embedding(6)
    -> Dense(16, L2) -> Dropout -> Dense(7) DELTA head
    forecast = last_known_value + predicted_delta

Loss: Huber.
Target: 7-day rolling mean of units_sold, log1p-transformed before scaling.
pandas .rolling(7) defaults to a trailing (non-centered) window, so there is
no look-ahead leakage in the smoothing itself.

Validation methodology:
Every split (CV folds, final training, lookback comparison, Optuna search)
is built PER CATEGORY and CHRONOLOGICALLY — each category's own earliest
days go to train, most recent days go to validation/test — before any
pooling across categories happens. Only the pooled TRAINING set is shuffled
afterward, for batch diversity. Validation order is never shuffled and
validation sequences are never drawn randomly from the full pool.

Final training uses a THREE-WAY split per category: train / early-stopping-
val / a held-out tail that is never shown to the model during training or
early stopping. That tail is what feature importance and prediction
intervals are computed on, so those numbers are genuinely out-of-sample.

(Baseline blending was removed from this file. It added a per-category grid
search, three new helper functions, and metadata fields, all just to mix in
a "repeat last value" persistence forecast — net effect was a lot of
surface area for not much gain once the leakage was fixed and most
categories were near a 20% MAPE ceiling anyway. If you want it back, the
git history has it.)
"""

import os
import sys
import json
import random
import logging
import warnings
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Global Seeds ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _seed_tensorflow():
    import tensorflow as tf
    tf.random.set_seed(SEED)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lstm")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "data" / "models"
PLOTS_DIR     = Path(__file__).parent.parent / "data" / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_CATEGORIES = [
    "bed_bath_table", "health_beauty", "sports_leisure",
    "furniture_decor", "computers_accessories", "housewares",
    "watches_gifts", "telephony", "garden_tools", "auto",
]

LOOKBACK        = 30
FORECAST_DAYS   = 7
EPOCHS          = 100
BATCH_SIZE      = 16
PATIENCE        = 15
N_FOLDS         = 3
TEST_DAYS       = 30
MIN_TRAIN_DAYS  = 120
EMBED_DIM       = 6   # rule of thumb: min(50, (n_categories+1)//2) = 5 for 10 cats

DEBUG_FOLDS         = False  # set from --debug-folds in main()
USE_LAYERNORM       = False  # set from --use-layernorm in main()
CANDIDATE_LOOKBACKS = (30, 60, 90)
PI_LEVELS           = (0.8, 0.95)

MODEL_HP_KEYS = {"lstm1", "lstm2", "dropout", "learning_rate"}
DEFAULT_HP = {"lstm1": 32, "lstm2": 16, "dropout": 0.4,
              "learning_rate": 0.001, "batch_size": BATCH_SIZE}


def split_hp(hp: dict) -> tuple[dict, int]:
    """Splits a hyperparameter dict into (model-building kwargs, batch_size)."""
    model_hp   = {k: v for k, v in hp.items() if k in MODEL_HP_KEYS}
    batch_size = hp.get("batch_size", BATCH_SIZE)
    return model_hp, batch_size


FEATURE_COLS = [
    "units_sold",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7",  "rolling_std_30",
    "seasonality_index",
    "month_sin", "month_cos",
    "dow_sin",   "dow_cos",
    "is_weekend", "is_holiday", "days_to_holiday",
    "trend_strength", "acceleration",
]
N_FEATURES = len(FEATURE_COLS)


# ── Metrics ─────────────────────────────────────────────────────────────────

def mape(actual, predicted):
    mask = actual > 0
    if mask.sum() < 5:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual, predicted):
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask  = denom > 0
    if mask.sum() < 5:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def mase(actual, predicted, train):
    """MASE in raw scale. `train` must already be inverse-transformed
    AND expm1'd — see to_raw_scale()."""
    naive_errors = np.abs(np.diff(train))
    if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / np.mean(naive_errors))


def forecast_skill(lstm_mape, baseline_mape):
    """% improvement of the LSTM over the baseline.
    Positive = LSTM better. Negative = LSTM worse."""
    if baseline_mape is None or baseline_mape == 0 or np.isnan(baseline_mape):
        return float("nan")
    return float((baseline_mape - lstm_mape) / baseline_mape * 100)


def all_metrics(actual, predicted, train=None):
    m = {
        "mape":  mape(actual, predicted),
        "smape": smape(actual, predicted),
        "rmse":  rmse(actual, predicted),
        "mae":   mae(actual, predicted),
        "bias":  float(np.mean(predicted - actual)),
    }
    if train is not None:
        m["mase"] = mase(actual, predicted, train)
    return m


def to_raw_scale(values: np.ndarray, target_scaler, clip_negative: bool = False) -> np.ndarray:
    """Undoes feature_engineering.py's np.log1p() -> MinMaxScaler transform.
    The model outputs a DELTA; the anchor is added back in scaled space
    BEFORE this function is called, so this always receives plain absolute
    scaled values.

    Set clip_negative=True for *predicted* values (model output can stray
    slightly negative/out-of-range). Leave False for ground-truth actuals.
    """
    flat = values.reshape(-1, 1)
    if clip_negative:
        flat = np.clip(flat, 0, None)
    inv = target_scaler.inverse_transform(flat).flatten()
    if clip_negative:
        inv = np.clip(inv, 0, None)
    raw = np.expm1(inv)
    return raw.reshape(values.shape)


# ── Data ──────────────────────────────────────────────────────────────────

def load_data():
    path = PROCESSED_DIR / "lstm_features.parquet"
    if not path.exists():
        log.error("lstm_features.parquet not found — run feature_engineering.py first")
        sys.exit(1)

    missing = [p for p in [
        PROCESSED_DIR / "target_scaler.pkl",
        PROCESSED_DIR / "minmax_scaler.pkl",
        PROCESSED_DIR / "feature_dates.parquet",
    ] if not p.exists()]
    if missing:
        log.error(f"Missing files: {[p.name for p in missing]}")
        sys.exit(1)

    df = pd.read_parquet(path)

    missing_cols = set(FEATURE_COLS) - set(df.columns)
    if missing_cols:
        log.error(f"Missing feature columns: {missing_cols}")
        log.error("  Re-run feature_engineering.py if this is an old parquet file.")
        sys.exit(1)

    dates_df = pd.read_parquet(PROCESSED_DIR / "feature_dates.parquet")
    df["sale_date"] = None
    for cat in df["category"].unique():
        mask      = df["category"] == cat
        cat_dates = dates_df[dates_df["category"] == cat]["sale_date"].values
        df.loc[mask, "sale_date"] = cat_dates[:mask.sum()]
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    log.info(f"Loaded {len(df):,} rows | {df['category'].nunique()} categories")
    return df


def load_scalers():
    return (
        joblib.load(PROCESSED_DIR / "target_scaler.pkl"),
        joblib.load(PROCESSED_DIR / "minmax_scaler.pkl"),
    )


def load_baseline_mapes() -> dict:
    """Each category's best baseline MAPE, for forecast_skill / beats_baseline."""
    path = PROCESSED_DIR / "baseline_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        summary = json.load(f)
    return {cat: v["best_baseline"] for cat, v in summary.items()
            if v.get("best_baseline") is not None}


def print_baseline_winners(baseline_mapes: dict) -> None:
    """Which baseline method (last_value / seasonal_naive / avg_demand /
    prophet) actually won per category, sorted by winning MAPE."""
    path = PROCESSED_DIR / "baseline_summary.json"
    if not path.exists():
        log.warning("  baseline_summary.json not found — skipping baseline-winner printout")
        return
    with open(path) as f:
        summary = json.load(f)

    log.info("  Baseline method that won, per category (sorted by winning MAPE):")
    rows = []
    for cat, v in summary.items():
        candidates = {k: val for k, val in v.items()
                      if k.endswith("_mape") and k != "lstm_target_mape"
                      and isinstance(val, (int, float))}
        if not candidates:
            continue
        best_method = min(candidates, key=candidates.get)
        rows.append((cat, best_method, candidates[best_method]))

    persistence_wins = 0
    for cat, method, val in sorted(rows, key=lambda r: r[2]):
        tag = ""
        if "last_value" in method or "seasonal_naive" in method:
            persistence_wins += 1
            tag = "  <- persistence-style"
        log.info(f"    {cat:<25} won by {method:<22} MAPE={val:.1f}%{tag}")

    if rows:
        log.info(f"  {persistence_wins}/{len(rows)} categories won by a persistence-style baseline.")


def print_level_bias(model, X_eval, id_eval, y_eval, anchor_eval, categories_eval,
                      target_scaler, fold_label: str = "fold 1") -> None:
    """Per-category mean(actual) vs mean(predicted), raw scale. Distinguishes
    a uniform level-shift bias from a variance/spike failure.
    """
    pred_delta = model.predict([X_eval, id_eval], verbose=0)
    pred_abs_scaled   = pred_delta + anchor_eval.reshape(-1, 1)
    actual_abs_scaled = y_eval

    actual_raw    = to_raw_scale(actual_abs_scaled, target_scaler, clip_negative=False)
    predicted_raw = to_raw_scale(pred_abs_scaled,   target_scaler, clip_negative=True)

    log.info(f"  Level bias check ({fold_label}, raw scale, mean across {FORECAST_DAYS}-day horizon):")

    for cid in sorted(set(categories_eval.tolist())):
        mask = categories_eval == cid
        if mask.sum() == 0:
            continue
        mean_actual = float(np.mean(actual_raw[mask]))
        mean_pred   = float(np.mean(predicted_raw[mask]))
        direction   = "UNDER" if mean_pred < mean_actual else "OVER"
        log.info(f"    cat_id={cid:<3} mean(actual)={mean_actual:>8.2f}  "
                 f"mean(predicted)={mean_pred:>8.2f}  ({direction} by "
                 f"{abs(mean_pred - mean_actual):.2f})")


def prepare_experiment_df(df: pd.DataFrame) -> pd.DataFrame:
    """7-day rolling mean smoothing of units_sold. .rolling(7) defaults to a
    trailing window, so this does not leak future values into the target."""
    df = df.copy()
    for cat in df["category"].unique():
        mask = df["category"] == cat
        df.loc[mask, "units_sold"] = (
            df.loc[mask, "units_sold"].rolling(7, min_periods=1).mean()
        )
    return df


def build_category_map(categories: list[str]) -> dict:
    """Deterministic category -> integer id mapping for the embedding layer."""
    return {cat: i for i, cat in enumerate(sorted(categories))}


def prepare_data_by_category(df: pd.DataFrame, categories: list[str],
                              lookback: int = LOOKBACK) -> dict[str, np.ndarray]:
    """{category: (n_days, N_FEATURES) array}, skipping categories without
    enough history for the given lookback."""
    out = {}
    for cat in categories:
        cat_df = (df[df["category"] == cat]
                  .sort_values("sale_date")
                  .reset_index(drop=True))
        n = len(cat_df)
        if n < MIN_TRAIN_DAYS + lookback + FORECAST_DAYS:
            log.warning(f"  {cat}: {n} rows insufficient for lookback={lookback} — skipping")
            continue
        out[cat] = cat_df[FEATURE_COLS].values.astype(np.float32)
    return out


# ── Sequence Builder ──────────────────────────────────────────────────────

def build_sequences(data: np.ndarray, lookback: int, forecast_days: int,
                     target_col: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (n_samples, lookback, n_features)
    y: (n_samples, forecast_days) — absolute scaled target values
    anchor: (n_samples,) — data[i-1, target_col], the last known value
        before the forecast window (same scaled space as y). The model
        predicts delta = y - anchor.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    min_len = lookback + forecast_days
    if len(data) < min_len:
        return (
            np.empty((0, lookback, data.shape[1]), dtype=np.float32),
            np.empty((0, forecast_days),           dtype=np.float32),
            np.empty((0,),                         dtype=np.float32),
        )

    X, y, anchor = [], [], []
    for i in range(lookback, len(data) - forecast_days + 1):
        X.append(data[i - lookback: i, :])
        y.append(data[i: i + forecast_days, target_col])
        anchor.append(data[i - 1, target_col])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(anchor, dtype=np.float32),
    )


def chronological_split_by_category(
    data_by_cat: dict[str, np.ndarray],
    cat_to_id: dict,
    lookback: int,
    val_frac: float = 0.20,
    holdout_end: "int | dict | None" = None,
) -> tuple[list, list, list, list, list, list, list, list]:
    """Per-category: build sequences, then split chronologically — earliest
    ~(1-val_frac) -> train, most recent ~val_frac -> val — before pooling.

    holdout_end truncates each category's series before the split:
      None -> full series, int -> same cutoff for all, dict -> per-category.

    Returns parallel lists: X_tr, y_tr, anchor_tr, id_tr, X_val, y_val,
    anchor_val, id_val (one entry per category with usable data).
    """
    X_tr_list,  y_tr_list,  anchor_tr_list,  id_tr_list  = [], [], [], []
    X_val_list, y_val_list, anchor_val_list, id_val_list = [], [], [], []

    for cat, data in data_by_cat.items():
        cid = cat_to_id[cat]

        if holdout_end is None:
            series = data
        elif isinstance(holdout_end, dict):
            if cat not in holdout_end:
                continue
            series = data[:holdout_end[cat]]
        else:
            series = data[:holdout_end]

        X, y, anchor = build_sequences(series, lookback, FORECAST_DAYS)
        if len(X) == 0:
            continue

        val_size = max(1, int(len(X) * val_frac))
        X_tr_list.append(X[:-val_size])
        y_tr_list.append(y[:-val_size])
        anchor_tr_list.append(anchor[:-val_size])
        id_tr_list.append(np.full(len(X) - val_size, cid, dtype=np.int32))

        X_val_list.append(X[-val_size:])
        y_val_list.append(y[-val_size:])
        anchor_val_list.append(anchor[-val_size:])
        id_val_list.append(np.full(val_size, cid, dtype=np.int32))

    return (X_tr_list, y_tr_list, anchor_tr_list, id_tr_list,
            X_val_list, y_val_list, anchor_val_list, id_val_list)


def chronological_split_three_way(
    data_by_cat: dict[str, np.ndarray],
    cat_to_id: dict,
    lookback: int,
    es_val_frac: float = 0.10,
    holdout_val_frac: float = 0.07,
) -> dict:
    """Per-category three-way chronological split: train / early-stopping-val
    (shown to Keras) / a held-out tail (never seen during training — used
    for feature importance and prediction intervals so those numbers are
    genuinely out-of-sample).

    Returns a dict of lists keyed "X_tr"/"y_tr"/"anchor_tr"/"id_tr",
    "X_es"/.../"id_es", "X_ho"/.../"id_ho" — concatenate by the caller.
    """
    out = {k: [] for k in [
        "X_tr", "y_tr", "anchor_tr", "id_tr",
        "X_es", "y_es", "anchor_es", "id_es",
        "X_ho", "y_ho", "anchor_ho", "id_ho",
    ]}
    for cat, data in data_by_cat.items():
        cid = cat_to_id[cat]
        X, y, anchor = build_sequences(data, lookback, FORECAST_DAYS)
        n = len(X)
        if n == 0:
            continue

        ho_size = max(1, int(n * holdout_val_frac))
        es_size = max(1, int(n * es_val_frac))
        if n - ho_size - es_size < 5:
            log.warning(f"  {cat}: insufficient sequences ({n}) for three-way split — skipping")
            continue

        X_ho, y_ho, anchor_ho = X[-ho_size:], y[-ho_size:], anchor[-ho_size:]
        X_es, y_es, anchor_es = (X[-ho_size - es_size:-ho_size],
                                  y[-ho_size - es_size:-ho_size],
                                  anchor[-ho_size - es_size:-ho_size])
        X_tr, y_tr, anchor_tr = (X[:-ho_size - es_size],
                                  y[:-ho_size - es_size],
                                  anchor[:-ho_size - es_size])

        out["X_tr"].append(X_tr);          out["y_tr"].append(y_tr)
        out["anchor_tr"].append(anchor_tr); out["id_tr"].append(np.full(len(X_tr), cid, dtype=np.int32))

        out["X_es"].append(X_es);          out["y_es"].append(y_es)
        out["anchor_es"].append(anchor_es); out["id_es"].append(np.full(len(X_es), cid, dtype=np.int32))

        out["X_ho"].append(X_ho);          out["y_ho"].append(y_ho)
        out["anchor_ho"].append(anchor_ho); out["id_ho"].append(np.full(len(X_ho), cid, dtype=np.int32))

    return out


def build_per_category_folds(data_by_cat: dict[str, np.ndarray],
                              lookback: int = LOOKBACK,
                              n_folds: int = N_FOLDS,
                              test_days: int = TEST_DAYS,
                              min_train_days: int = MIN_TRAIN_DAYS) -> dict[str, list]:
    """Walk-forward fold boundaries computed independently per category,
    using each category's own series length. Mirrors baseline_models.py.

    Returns {category: [(train_end, test_start, test_end, fold_num), ...]}
    """
    per_cat_folds = {}
    for cat, data in data_by_cat.items():
        n = len(data)
        folds = []
        for i in range(n_folds):
            test_end   = n - (n_folds - i - 1) * test_days
            test_start = test_end - test_days
            train_end  = test_start
            if train_end < min_train_days + lookback:
                continue
            if test_start >= test_end or test_end > n:
                continue
            folds.append((train_end, test_start, test_end, i + 1))
        per_cat_folds[cat] = folds
    return per_cat_folds


def compute_smoothed_baseline_mapes(data_by_cat: dict[str, np.ndarray],
                                     target_scaler, lookback: int = LOOKBACK,
                                     n_folds: int = N_FOLDS,
                                     test_days: int = TEST_DAYS,
                                     seasonal_period: int = 7) -> dict[str, dict]:
    """Apples-to-apples baselines: last_value / seasonal_naive / avg_demand,
    computed on the SAME smoothed target the LSTM predicts, walking forward
    through the test window exactly the way build_sequences()'s sliding
    anchor does (not one stale anchor for the whole window).

    Returns {category: {"last_value_mape", "seasonal_naive_mape",
                         "avg_demand_mape", "best_baseline",
                         "best_baseline_method"}}
    """
    per_cat_folds = build_per_category_folds(data_by_cat, lookback=lookback,
                                              n_folds=n_folds, test_days=test_days)
    out = {}

    for cat, folds in per_cat_folds.items():
        data = data_by_cat[cat]
        fold_method_mapes = defaultdict(list)

        for train_end, test_start, test_end, fold_num in folds:
            full_raw  = to_raw_scale(data[:test_end, 0], target_scaler, clip_negative=False)
            train_avg = float(np.mean(full_raw[:train_end]))

            last_value_preds, seasonal_preds, avg_preds, actuals = [], [], [], []

            for i in range(test_start, test_end - FORECAST_DAYS + 1):
                anchor_idx = i - 1
                if anchor_idx < 0:
                    continue
                y_true = full_raw[i: i + FORECAST_DAYS]
                if len(y_true) < FORECAST_DAYS:
                    continue

                last_value_preds.append(np.full(FORECAST_DAYS, full_raw[anchor_idx]))

                season_window = []
                for d in range(FORECAST_DAYS):
                    s_idx = i + d - seasonal_period
                    season_window.append(full_raw[s_idx] if s_idx >= 0 else full_raw[anchor_idx])
                seasonal_preds.append(np.array(season_window))

                avg_preds.append(np.full(FORECAST_DAYS, train_avg))
                actuals.append(y_true)

            if not actuals:
                continue

            actuals_arr = np.array(actuals)
            fold_method_mapes["last_value_mape"].append(
                mape(actuals_arr.flatten(), np.array(last_value_preds).flatten()))
            fold_method_mapes["seasonal_naive_mape"].append(
                mape(actuals_arr.flatten(), np.array(seasonal_preds).flatten()))
            fold_method_mapes["avg_demand_mape"].append(
                mape(actuals_arr.flatten(), np.array(avg_preds).flatten()))

        if not fold_method_mapes:
            continue

        method_means = {k: float(np.nanmean(v)) for k, v in fold_method_mapes.items()}
        best_method = min(method_means, key=method_means.get)
        out[cat] = {**method_means,
                    "best_baseline": method_means[best_method],
                    "best_baseline_method": best_method}

    return out


def print_smoothed_baseline_winners(smoothed_baselines: dict) -> None:
    if not smoothed_baselines:
        log.warning("  No smoothed baselines computed — skipping printout")
        return
    log.info("  Smoothed-target baseline winners (apples-to-apples with the LSTM, "
             "sorted by winning MAPE):")
    rows = sorted(smoothed_baselines.items(), key=lambda kv: kv[1]["best_baseline"])
    persistence_wins = 0
    for cat, v in rows:
        method = v["best_baseline_method"]
        tag = ""
        if "last_value" in method or "seasonal_naive" in method:
            persistence_wins += 1
            tag = "  <- persistence-style"
        log.info(f"    {cat:<25} won by {method:<22} MAPE={v['best_baseline']:.1f}%{tag}")
    log.info(f"  {persistence_wins}/{len(rows)} categories won by a persistence-style baseline.")


# ── Model ───────────────────────────────────────────────────────────────────

def build_global_model(n_categories: int, lookback: int = LOOKBACK,
                        lstm1: int = 32, lstm2: int = 16, dropout: float = 0.4,
                        learning_rate: float = 0.001, embed_dim: int = EMBED_DIM) -> "tf.keras.Model":
    """Global model: one shared LSTM stack across all categories, fused with
    a learned per-category embedding before the output head.

    - sequence branch: LSTM(lstm1) -> Norm -> Dropout -> LSTM(lstm2) -> Dropout
    - category branch: Embedding(n_categories, embed_dim) -> Flatten
    - head: Concatenate -> Dense(16, relu, L2) -> Dropout -> Dense(7) [delta]
    - Huber loss, clipnorm=1.0, L2(1e-4) on LSTM kernels and Dense(16)

    Output is a DELTA from the last known value. The caller adds the anchor
    back (in scaled space) before inverse-transforming.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                          BatchNormalization, LayerNormalization,
                                          Embedding, Flatten, Concatenate)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    seq_input = Input(shape=(lookback, N_FEATURES), name="sequence_input")
    cat_input = Input(shape=(1,), name="category_input")

    x = LSTM(lstm1, return_sequences=True, kernel_regularizer=l2(1e-4))(seq_input)
    x = LayerNormalization()(x) if USE_LAYERNORM else BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = LSTM(lstm2, return_sequences=False, kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout)(x)

    cat_embed = Embedding(input_dim=n_categories, output_dim=embed_dim,
                          name="category_embedding")(cat_input)
    cat_embed = Flatten()(cat_embed)

    merged = Concatenate()([x, cat_embed])
    merged = Dense(16, activation="relu", kernel_regularizer=l2(1e-4))(merged)
    merged = Dropout(dropout)(merged)
    output = Dense(FORECAST_DAYS, name="delta_output")(merged)

    model = Model(inputs=[seq_input, cat_input], outputs=output,
                 name="lstm_global_demand_forecaster")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


def get_callbacks():
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    return [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                           patience=7, min_lr=1e-6, verbose=0),
    ]


def check_overfitting(history, label: str, experiment: str) -> dict:
    """overfit_ratio computed AT best_epoch (the epoch EarlyStopping actually
    restored weights from), not over a trailing window."""
    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]

    best_epoch    = int(np.argmin(val_loss)) + 1
    best_val_loss = float(min(val_loss))
    overfit_ratio = float(val_loss[best_epoch - 1] / (train_loss[best_epoch - 1] + 1e-9))
    is_overfit    = overfit_ratio > 1.5

    if is_overfit:
        log.warning(f"  ⚠ OVERFITTING: ratio={overfit_ratio:.2f} at best epoch "
                    f"{best_epoch}/{len(train_loss)}")
    else:
        log.info(f"  No overfitting — best epoch {best_epoch}, "
                 f"val_loss={best_val_loss:.5f}, ratio={overfit_ratio:.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, label="Train",      color="steelblue")
    ax.plot(val_loss,   label="Validation", color="orange")
    ax.axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.7,
               label=f"Best ({best_epoch})")
    ax.set_title(f"Training — {label} [{experiment}]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"training_{experiment}_{label}.png", dpi=100)
    plt.close()

    return {
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "overfit_ratio": overfit_ratio,
        "is_overfit":    is_overfit,
        "total_epochs":  len(train_loss),
    }


def permutation_feature_importance(model, X_eval, id_eval, y_eval, anchor_eval, target_scaler,
                                    feature_names=FEATURE_COLS, n_repeats: int = 5,
                                    seed: int = 42) -> pd.DataFrame:
    """For each feature channel, shuffle that feature's lookback window
    across samples and measure the resulting MAPE increase vs baseline.
    Averaged over n_repeats shuffles."""
    rng = np.random.RandomState(seed)

    def predict_mape(X_batch):
        pred_delta    = model.predict([X_batch, id_eval], verbose=0)
        pred_abs      = pred_delta + anchor_eval.reshape(-1, 1)
        actual_raw    = to_raw_scale(y_eval,  target_scaler, clip_negative=False)
        predicted_raw = to_raw_scale(pred_abs, target_scaler, clip_negative=True)
        return mape(actual_raw.flatten(), predicted_raw.flatten())

    baseline_mape_val = predict_mape(X_eval)
    rows = []
    for f_idx, f_name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, :, f_idx] = X_perm[perm_idx, :, f_idx]
            deltas.append(predict_mape(X_perm) - baseline_mape_val)
        rows.append({
            "feature":           f_name,
            "mape_increase":     float(np.mean(deltas)),
            "mape_increase_std": float(np.std(deltas)),
        })

    out = pd.DataFrame(rows).sort_values("mape_increase", ascending=False).reset_index(drop=True)
    out["baseline_mape"] = baseline_mape_val
    return out


def plot_feature_importance(importance_df: pd.DataFrame, experiment: str, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance_df.head(15).iloc[::-1]
    ax.barh(top["feature"], top["mape_increase"], color="steelblue")
    ax.set_xlabel("MAPE increase when shuffled (pp)")
    ax.set_title(f"Permutation Feature Importance — {label} [{experiment}]")
    plt.tight_layout()
    path = PLOTS_DIR / f"feature_importance_{experiment}_{label}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def compute_prediction_intervals(residuals: np.ndarray, levels=PI_LEVELS) -> dict:
    """residuals: (n_samples, FORECAST_DAYS) array of (actual_raw - predicted_raw)
    from the held-out slice."""
    out = {}
    for day in range(residuals.shape[1]):
        col = residuals[:, day]
        col = col[~np.isnan(col)]
        day_out = {}
        for level in levels:
            alpha = 1 - level
            lower = float(np.quantile(col, alpha / 2))     if len(col) else 0.0
            upper = float(np.quantile(col, 1 - alpha / 2))  if len(col) else 0.0
            day_out[str(int(level * 100))] = {"lower_offset": lower, "upper_offset": upper}
        out[str(day + 1)] = day_out
    return out


def apply_prediction_intervals(point_forecast: np.ndarray, intervals: dict,
                                level: int = 95) -> tuple[np.ndarray, np.ndarray]:
    """Returns (lower_bound, upper_bound) for a serving endpoint."""
    lower = np.array([point_forecast[d] + intervals[str(d + 1)][str(level)]["lower_offset"]
                       for d in range(len(point_forecast))])
    upper = np.array([point_forecast[d] + intervals[str(d + 1)][str(level)]["upper_offset"]
                       for d in range(len(point_forecast))])
    return np.clip(lower, 0, None), np.clip(upper, 0, None)


def flatten_per_category_metrics(per_cat_metrics: dict) -> dict:
    flat = {}
    for cat, m in per_cat_metrics.items():
        for k, v in m.items():
            flat[f"{cat}__{k}"] = v
    return flat


def log_mlflow(run_label: str, experiment: str, params: dict, metrics: dict,
               artifact_paths: "list[str] | None" = None,
               mlflow_experiment: "str | None" = None):
    """Fails loudly — silent failures hide broken tracking."""
    try:
        import mlflow
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(mlflow_experiment or f"supply_chain_lstm_{experiment}")

        with mlflow.start_run(run_name=f"{experiment}_{run_label}"):
            mlflow.log_params(params)
            clean_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None
                and not (isinstance(v, float) and np.isnan(v))
            }
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)
            for p in (artifact_paths or []):
                if Path(p).exists():
                    mlflow.log_artifact(str(p))

        log.info(f"  MLflow → {uri} | {experiment}_{run_label}")
    except Exception as e:
        log.error(f"  MLflow FAILED: {type(e).__name__}: {e}")
        log.error("  Start with: mlflow ui --port 5000")


# ── Walk-Forward CV (global model) ───────────────────────────────────────

def walk_forward_cv_global(data_by_cat: dict[str, np.ndarray], cat_to_id: dict,
                            target_scaler, experiment: str, lookback: int = LOOKBACK,
                            hp: "dict | None" = None,
                            print_diagnostics: bool = True) -> tuple[dict, dict]:
    """One shared model per fold. Pools training sequences across all
    categories (with a parallel category-id array), trains a single model,
    evaluates per category AND pooled. Fold boundaries are per category, so
    each category is tested on its own true most-recent TEST_DAYS per fold.

    Returns (per_category_metrics: {cat: metrics_dict}, pooled_metrics: dict).
    """
    hp = hp or DEFAULT_HP
    model_hp, batch_size = split_hp(hp)

    per_cat_folds = build_per_category_folds(data_by_cat, lookback=lookback)
    if not any(per_cat_folds.values()):
        log.warning("  No valid folds for global model")
        return {}, {}

    per_cat_fold_metrics = defaultdict(list)
    pooled_fold_metrics  = []
    diagnostics_printed  = False

    for fold_num in range(1, N_FOLDS + 1):
        holdout_ends = {}
        test_windows = {}
        for cat, folds in per_cat_folds.items():
            this_fold = next((f for f in folds if f[3] == fold_num), None)
            if this_fold is None:
                continue
            train_end, test_start, test_end, _ = this_fold
            holdout_ends[cat] = train_end
            test_windows[cat] = (train_end, test_start, test_end)

        if not holdout_ends:
            log.warning(f"  Fold {fold_num}: no categories had enough history — skipping")
            continue

        (X_tr_list, y_tr_list, anchor_tr_list, id_tr_list,
         X_val_list, y_val_list, anchor_val_list, id_val_list) = (
            chronological_split_by_category(data_by_cat, cat_to_id, lookback,
                                             val_frac=0.20, holdout_end=holdout_ends)
        )

        test_by_cat = {}
        for cat, (train_end, test_start, test_end) in test_windows.items():
            data = data_by_cat[cat]
            test_data = data[test_start - lookback: test_end]
            X_test, y_test, anchor_test = build_sequences(test_data, lookback, FORECAST_DAYS)
            if len(X_test) == 0:
                continue
            test_by_cat[cat] = (X_test, y_test, anchor_test, data[:train_end])

        if not X_tr_list:
            log.warning(f"  Fold {fold_num}: no categories had usable train data — skipping")
            continue

        X_val      = np.concatenate(X_val_list)
        y_val       = np.concatenate(y_val_list)
        anchor_val  = np.concatenate(anchor_val_list)
        id_val      = np.concatenate(id_val_list)

        X_train_all      = np.concatenate(X_tr_list)
        y_train_all       = np.concatenate(y_tr_list)
        anchor_train_all  = np.concatenate(anchor_tr_list)
        id_train_all      = np.concatenate(id_tr_list)
        perm = np.random.RandomState(SEED + fold_num).permutation(len(X_train_all))
        X_tr, y_tr, anchor_tr, id_tr = (X_train_all[perm], y_train_all[perm],
                                         anchor_train_all[perm], id_train_all[perm])

        delta_tr  = y_tr  - anchor_tr.reshape(-1, 1)
        delta_val = y_val - anchor_val.reshape(-1, 1)

        model = build_global_model(len(cat_to_id), lookback=lookback, **model_hp)
        model.fit([X_tr, id_tr], delta_tr, epochs=EPOCHS, batch_size=batch_size,
                  validation_data=([X_val, id_val], delta_val),
                  callbacks=get_callbacks(), verbose=0)

        if print_diagnostics and not diagnostics_printed:
            print_level_bias(model, X_val, id_val, y_val, anchor_val, id_val,
                             target_scaler, fold_label=f"fold {fold_num} (val set)")
            diagnostics_printed = True

        fold_actual_all, fold_pred_all, fold_train_all = [], [], []
        for cat, (X_test, y_test, anchor_test, train_data) in test_by_cat.items():
            cid     = cat_to_id[cat]
            id_test = np.full(len(X_test), cid, dtype=np.int32)
            pred_delta = model.predict([X_test, id_test], verbose=0)
            y_pred     = pred_delta + anchor_test.reshape(-1, 1)

            actual_raw    = to_raw_scale(y_test, target_scaler, clip_negative=False)
            predicted_raw = to_raw_scale(y_pred, target_scaler, clip_negative=True)
            train_raw     = to_raw_scale(train_data[:, 0], target_scaler, clip_negative=False)

            if DEBUG_FOLDS:
                log.info(f"  [debug] {cat} fold {fold_num}: "
                         f"pred range [{np.min(predicted_raw):.1f}, {np.max(predicted_raw):.1f}]  "
                         f"actual range [{np.min(actual_raw):.1f}, {np.max(actual_raw):.1f}]")

            m = all_metrics(actual_raw.flatten(), predicted_raw.flatten(), train=train_raw)
            per_cat_fold_metrics[cat].append(m)

            fold_actual_all.append(actual_raw.flatten())
            fold_pred_all.append(predicted_raw.flatten())
            fold_train_all.append(train_raw)

        pooled_actual = np.concatenate(fold_actual_all)
        pooled_pred  = np.concatenate(fold_pred_all)
        pooled_train = np.concatenate(fold_train_all)
        pooled_m = all_metrics(pooled_actual, pooled_pred, train=pooled_train)
        pooled_fold_metrics.append(pooled_m)          # <-- ADD THIS LINE BACK

        log.info(f"  Fold {fold_num} ({len(test_by_cat)} categories, each tested on "
         f"its OWN last {TEST_DAYS}d): "
         f"Pooled MAPE={pooled_m['mape']:.1f}%  MASE={pooled_m['mase']:.2f}  "
         f"Bias={pooled_m['bias']:+.2f}")
        
    per_cat_summary = {
        cat: {k: float(np.nanmean([f[k] for f in fl])) for k in fl[0]}
        for cat, fl in per_cat_fold_metrics.items() if fl
    }
    pooled_summary = (
        {k: float(np.nanmean([f[k] for f in pooled_fold_metrics])) for k in pooled_fold_metrics[0]}
        if pooled_fold_metrics else {}
    )
    return per_cat_summary, pooled_summary


def train_final_global(data_by_cat: dict[str, np.ndarray], cat_to_id: dict,
                        target_scaler, experiment: str, lookback: int = LOOKBACK,
                        hp: "dict | None" = None):
    """Three-way per-category chronological split: train / early-stopping-val
    (shown to Keras) / a held-out tail (never seen during training — used
    for feature importance and prediction intervals)."""
    hp = hp or DEFAULT_HP
    model_hp, batch_size = split_hp(hp)

    split = chronological_split_three_way(data_by_cat, cat_to_id, lookback)
    if not split["X_tr"]:
        log.error("  No sequences available for global final training")
        return {}, None, (None, None, None, None)

    X_es      = np.concatenate(split["X_es"]);      y_es      = np.concatenate(split["y_es"])
    anchor_es = np.concatenate(split["anchor_es"]);  id_es     = np.concatenate(split["id_es"])

    X_ho      = np.concatenate(split["X_ho"]);      y_ho      = np.concatenate(split["y_ho"])
    anchor_ho = np.concatenate(split["anchor_ho"]);  id_ho     = np.concatenate(split["id_ho"])

    X_all      = np.concatenate(split["X_tr"]);     y_all      = np.concatenate(split["y_tr"])
    anchor_all = np.concatenate(split["anchor_tr"]); id_all    = np.concatenate(split["id_tr"])
    perm = np.random.RandomState(SEED).permutation(len(X_all))
    X_tr, y_tr, anchor_tr, id_tr = X_all[perm], y_all[perm], anchor_all[perm], id_all[perm]

    delta_tr = y_tr - anchor_tr.reshape(-1, 1)
    delta_es = y_es - anchor_es.reshape(-1, 1)

    model   = build_global_model(len(cat_to_id), lookback=lookback, **model_hp)
    history = model.fit([X_tr, id_tr], delta_tr, epochs=EPOCHS, batch_size=batch_size,
                        validation_data=([X_es, id_es], delta_es),
                        callbacks=get_callbacks(), verbose=0)

    overfit_info = check_overfitting(history, "global", experiment)

    # The held-out tail (X_ho etc.) is what feature importance and
    # prediction intervals are computed on downstream — it's never seen
    # during training or early stopping, so it's genuinely out-of-sample.
    X_val, y_val, anchor_val, id_val = X_ho, y_ho, anchor_ho, id_ho

    model_path = MODELS_DIR / f"lstm_global_{experiment}.h5"
    model.save(str(model_path))

    with open(MODELS_DIR / f"lstm_global_{experiment}_categories.json", "w") as f:
        json.dump(cat_to_id, f, indent=2)

    metadata = {
        "mode":            "global",
        "experiment":      experiment,
        "lookback":        lookback,
        "forecast_days":   FORECAST_DAYS,
        "n_features":      N_FEATURES,
        "feature_cols":    FEATURE_COLS,
        "categories":      list(cat_to_id.keys()),
        "n_categories":    len(cat_to_id),
        "embed_dim":       EMBED_DIM,
        "hyperparameters": {**model_hp, "batch_size": batch_size},
        "architecture":    (f"Global: LSTM{model_hp['lstm1']}-Norm-Dropout-"
                            f"LSTM{model_hp['lstm2']}-Dropout-"
                            f"[+CategoryEmbedding{EMBED_DIM}]-Dense16(L2)-Dropout-Dense7(delta)"),
        "normalization":   "layer" if USE_LAYERNORM else "batch",
        "output_type":     "delta_from_last_known_value",
        "loss":            "huber",
        "log_transform":   True,
        "val_split":       "three_way: train / early_stopping_val(0.10) / held_out(0.07)",
        **overfit_info,
    }
    with open(MODELS_DIR / f"lstm_global_{experiment}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"  Saved → {model_path.name}")
    return overfit_info, model, (X_val, y_val, anchor_val, id_val)


# ── Lookback Tuning ─────────────────────────────────────────────────────

def compare_lookbacks(data_by_cat: dict[str, np.ndarray], cat_to_id: dict, target_scaler,
                       candidates=CANDIDATE_LOOKBACKS, hp: "dict | None" = None) -> tuple[int, dict]:
    """Single most-recent-fold comparison across candidate lookbacks
    (each category's own true last TEST_DAYS), reduced epoch budget.
    Not a full walk-forward grid search — re-run full CV with the winner."""
    hp = hp or DEFAULT_HP
    model_hp, batch_size = split_hp(hp)
    results = {}

    for lb in candidates:
        holdout_ends = {}
        test_windows = {}
        for cat, data in data_by_cat.items():
            n          = len(data)
            test_start = n - TEST_DAYS
            train_end  = test_start
            if train_end < MIN_TRAIN_DAYS + lb:
                continue
            holdout_ends[cat] = train_end
            test_windows[cat] = (train_end, test_start, n)

        if not holdout_ends:
            log.warning(f"  lookback={lb}: insufficient data — skipping")
            continue

        (X_tr_list, y_tr_list, anchor_tr_list, id_tr_list,
         X_val_list, y_val_list, anchor_val_list, id_val_list) = (
            chronological_split_by_category(data_by_cat, cat_to_id, lb,
                                             val_frac=0.20, holdout_end=holdout_ends)
        )

        X_te_list, y_te_list, anchor_te_list, id_te_list = [], [], [], []
        for cat, (train_end, test_start, test_end) in test_windows.items():
            cid       = cat_to_id[cat]
            data      = data_by_cat[cat]
            test_data = data[test_start - lb: test_end]
            X_test, y_test, anchor_test = build_sequences(test_data, lb, FORECAST_DAYS)
            if len(X_test) == 0:
                continue
            X_te_list.append(X_test)
            y_te_list.append(y_test)
            anchor_te_list.append(anchor_test)
            id_te_list.append(np.full(len(X_test), cid, dtype=np.int32))

        if not X_tr_list or not X_te_list:
            log.warning(f"  lookback={lb}: no categories had usable data — skipping")
            continue

        X_train_all      = np.concatenate(X_tr_list)
        y_train_all       = np.concatenate(y_tr_list)
        anchor_train_all  = np.concatenate(anchor_tr_list)
        id_train_all      = np.concatenate(id_tr_list)
        X_val      = np.concatenate(X_val_list)
        y_val       = np.concatenate(y_val_list)
        anchor_val  = np.concatenate(anchor_val_list)
        id_val      = np.concatenate(id_val_list)
        X_test_all      = np.concatenate(X_te_list)
        y_test_all       = np.concatenate(y_te_list)
        anchor_test_all  = np.concatenate(anchor_te_list)
        id_test_all      = np.concatenate(id_te_list)

        perm = np.random.RandomState(SEED).permutation(len(X_train_all))
        X_tr, y_tr, anchor_tr, id_tr = (X_train_all[perm], y_train_all[perm],
                                         anchor_train_all[perm], id_train_all[perm])

        delta_tr  = y_tr  - anchor_tr.reshape(-1, 1)
        delta_val = y_val - anchor_val.reshape(-1, 1)

        model = build_global_model(len(cat_to_id), lookback=lb, **model_hp)
        model.fit([X_tr, id_tr], delta_tr, validation_data=([X_val, id_val], delta_val),
                  epochs=min(EPOCHS, 60), batch_size=batch_size,
                  callbacks=get_callbacks(), verbose=0)

        pred_delta = model.predict([X_test_all, id_test_all], verbose=0)
        y_pred     = pred_delta + anchor_test_all.reshape(-1, 1)
        actual_raw    = to_raw_scale(y_test_all, target_scaler, clip_negative=False)
        predicted_raw = to_raw_scale(y_pred,     target_scaler, clip_negative=True)
        pooled_mape   = mape(actual_raw.flatten(), predicted_raw.flatten())
        results[lb]   = pooled_mape
        log.info(f"  lookback={lb:>3}d → pooled MAPE={pooled_mape:.2f}% "
                 f"({len(test_windows)} categories, each on its own last {TEST_DAYS}d)")

    if not results:
        log.warning("  No valid lookback candidates evaluated — keeping default")
        return LOOKBACK, {}

    best_lb = min(results, key=results.get)
    log.info(f"  Best lookback: {best_lb}d (MAPE={results[best_lb]:.2f}%)")
    with open(PROCESSED_DIR / "lookback_comparison.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    return best_lb, results


# ── Hyperparameter Tuning (Optuna) ───────────────────────────────────────

def run_optuna_search(data_by_cat: dict[str, np.ndarray], cat_to_id: dict, target_scaler,
                       lookback: int = LOOKBACK, n_trials: int = 20) -> dict:
    """Searches lstm1/lstm2/dropout/learning_rate/batch_size on a single
    train/val split (each category's own true most-recent TEST_DAYS held
    out). Full walk-forward CV is run afterward with the winning hyperparams."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("optuna not installed — run: pip install optuna. Using default hyperparameters.")
        return DEFAULT_HP

    holdout_ends = {}
    for cat, data in data_by_cat.items():
        n          = len(data)
        test_start = n - TEST_DAYS
        train_end  = test_start
        if train_end < MIN_TRAIN_DAYS + lookback:
            continue
        holdout_ends[cat] = train_end

    (X_tr_list, y_tr_list, anchor_tr_list, id_tr_list,
     X_val_list, y_val_list, anchor_val_list, id_val_list) = (
        chronological_split_by_category(data_by_cat, cat_to_id, lookback,
                                         val_frac=0.20, holdout_end=holdout_ends)
    )

    if not X_tr_list:
        log.warning("  No data available for HP search — using defaults")
        return DEFAULT_HP

    X_train_all      = np.concatenate(X_tr_list)
    y_train_all       = np.concatenate(y_tr_list)
    anchor_train_all  = np.concatenate(anchor_tr_list)
    X_val      = np.concatenate(X_val_list)
    y_val       = np.concatenate(y_val_list)
    anchor_val  = np.concatenate(anchor_val_list)
    id_val      = np.concatenate(id_val_list)

    perm = np.random.RandomState(SEED).permutation(len(X_train_all))
    X_tr, y_tr, anchor_tr = X_train_all[perm], y_train_all[perm], anchor_train_all[perm]
    id_tr = np.concatenate(id_tr_list)[perm]

    delta_tr  = y_tr  - anchor_tr.reshape(-1, 1)
    delta_val = y_val - anchor_val.reshape(-1, 1)

    def objective(trial):
        hp = {
            "lstm1":         trial.suggest_categorical("lstm1", [16, 32, 48]),
            "lstm2":         trial.suggest_categorical("lstm2", [8, 16, 24]),
            "dropout":       trial.suggest_float("dropout", 0.2, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [8, 16, 32]),
        }
        model_hp, batch_size = split_hp(hp)
        model = build_global_model(len(cat_to_id), lookback=lookback, **model_hp)
        history = model.fit([X_tr, id_tr], delta_tr, validation_data=([X_val, id_val], delta_val),
                            epochs=min(EPOCHS, 60), batch_size=batch_size,
                            callbacks=get_callbacks(), verbose=0)
        return min(history.history["val_loss"])

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info(f"  Optuna best params: {study.best_params} | best val_loss={study.best_value:.5f}")
    with open(PROCESSED_DIR / "optuna_best_params.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value,
                  "n_trials": n_trials}, f, indent=2)

    return study.best_params


# ── Experiment Runner ─────────────────────────────────────────────────────

def run_experiment_global(df: pd.DataFrame, experiment: str, target_scaler,
                           baseline_mapes: dict, skip_cv: bool, lookback: int, hp: dict,
                           compute_importance: bool, compute_intervals: bool) -> list[dict]:
    log.info(f"\n{'='*55}")
    log.info(f"EXPERIMENT {experiment.upper()} — GLOBAL MODEL (anchor/delta head)")
    log.info(f"Target: 7-day rolling mean (smoothed demand, log-transformed) | "
             f"Norm: {'LayerNorm' if USE_LAYERNORM else 'BatchNorm'}")
    log.info(f"{'='*55}")

    t0 = time.time()
    exp_df      = prepare_experiment_df(df)
    data_by_cat = prepare_data_by_category(exp_df, TOP_CATEGORIES, lookback=lookback)

    log.info("  Category series lengths (days):")
    for cat, data in sorted(data_by_cat.items(), key=lambda kv: -len(kv[1])):
        log.info(f"    {cat:<25} {len(data)}")

    if not data_by_cat:
        log.error("  No categories had sufficient data — skipping experiment")
        return []

    cat_to_id = build_category_map(list(data_by_cat.keys()))
    log.info(f"  Pooling {len(data_by_cat)} categories into 1 global model")

    print_baseline_winners(baseline_mapes)

    smoothed_baselines = compute_smoothed_baseline_mapes(data_by_cat, target_scaler, lookback=lookback)
    print_smoothed_baseline_winners(smoothed_baselines)
    baseline_mapes = {cat: v["best_baseline"] for cat, v in smoothed_baselines.items()}

    cv_per_cat, cv_pooled = {}, {}
    if not skip_cv:
        log.info(f"  Walk-Forward CV ({N_FOLDS} folds, 1 model fit per fold, "
                 f"each category tested on its OWN last {TEST_DAYS}d):")
        cv_per_cat, cv_pooled = walk_forward_cv_global(
            data_by_cat, cat_to_id, target_scaler, experiment,
            lookback=lookback, hp=hp, print_diagnostics=True)
        if cv_pooled.get("mape") is not None:
            log.info(f"  Pooled CV → MAPE={cv_pooled['mape']:.1f}%  "
                     f"MASE={cv_pooled.get('mase', 0):.2f}  "
                     f"Bias={cv_pooled.get('bias', 0):+.2f}")
        if cv_per_cat:
            log.info("  Per-category CV MAPE (sorted, worst first) vs each "
                     "category's own best baseline:")
            for cat, m in sorted(cv_per_cat.items(), key=lambda kv: -kv[1].get("mape", 0)):
                base     = baseline_mapes.get(cat)
                base_str = f"{base:.1f}%" if base is not None else "n/a"
                beats    = (base is not None and m.get("mape") is not None
                            and m["mape"] < base)
                mark     = "✓" if beats else "✗"
                log.info(f"    {mark} {cat:<25} LSTM={m.get('mape', float('nan')):>6.1f}%   "
                         f"baseline={base_str:>8}")

    log.info("  Final global model training:")
    overfit_info, model, (X_val, y_val, anchor_val, id_val) = train_final_global(
        data_by_cat, cat_to_id, target_scaler, experiment, lookback=lookback, hp=hp)

    if model is not None:
        print_level_bias(model, X_val, id_val, y_val, anchor_val, id_val,
                         target_scaler, fold_label="final model (held-out tail)")

    artifact_paths = [str(MODELS_DIR / f"lstm_global_{experiment}.h5")]

    if compute_intervals and model is not None:
        y_val_pred_delta = model.predict([X_val, id_val], verbose=0)
        y_val_pred        = y_val_pred_delta + anchor_val.reshape(-1, 1)
        actual_raw    = to_raw_scale(y_val,      target_scaler, clip_negative=False)
        predicted_raw = to_raw_scale(y_val_pred, target_scaler, clip_negative=True)

        residuals = actual_raw - predicted_raw
        pi = compute_prediction_intervals(residuals)
        pi_path = PROCESSED_DIR / f"lstm_global_{experiment}_intervals.json"
        with open(pi_path, "w") as f:
            json.dump(pi, f, indent=2)
        artifact_paths.append(str(pi_path))
        for level in PI_LEVELS:
            lvl = str(int(level * 100))
            avg_width = float(np.mean([
                pi[str(d + 1)][lvl]["upper_offset"] - pi[str(d + 1)][lvl]["lower_offset"]
                for d in range(FORECAST_DAYS)
            ]))
            log.info(f"  {lvl}% prediction interval — avg width ±{avg_width / 2:.1f} units")

    if compute_importance and model is not None:
        log.info("  Computing permutation feature importance...")
        importance_df = permutation_feature_importance(
            model, X_val, id_val, y_val, anchor_val, target_scaler)
        imp_path = PROCESSED_DIR / f"feature_importance_global_{experiment}.csv"
        importance_df.to_csv(imp_path, index=False)
        plot_feature_importance(importance_df, experiment, "global")
        artifact_paths.append(str(imp_path))
        top5 = ", ".join(importance_df.head(5)["feature"].tolist())
        log.info(f"  Top 5 features by importance: {top5}")

    skill_scores = {
        cat: forecast_skill(m.get("mape", float("nan")), baseline_mapes.get(cat))
        for cat, m in cv_per_cat.items()
    }
    valid_skills = [v for v in skill_scores.values() if not np.isnan(v)]
    avg_skill    = float(np.mean(valid_skills)) if valid_skills else float("nan")

    model_hp, batch_size = split_hp(hp)
    mlflow_params = {
        "mode": "global", "experiment": experiment, "lookback": lookback,
        "forecast_days": FORECAST_DAYS, "n_features": N_FEATURES,
        "n_categories": len(cat_to_id), "embed_dim": EMBED_DIM,
        "loss": "huber", "log_transform": True, "output_type": "delta",
        "normalization": "layer" if USE_LAYERNORM else "batch",
        "batch_size": batch_size, **{f"hp_{k}": v for k, v in model_hp.items()},
    }
    mlflow_metrics = {
        **{f"pooled_{k}": v for k, v in cv_pooled.items()},
        **flatten_per_category_metrics(cv_per_cat),
        **{f"skill_{cat}": v for cat, v in skill_scores.items() if not np.isnan(v)},
        "avg_forecast_skill_pct": avg_skill,
        "best_val_loss":  overfit_info.get("best_val_loss", 0),
        "overfit_ratio":  overfit_info.get("overfit_ratio", 0),
        "best_epoch":     overfit_info.get("best_epoch", 0),
    }
    log_mlflow("global", experiment, mlflow_params, mlflow_metrics, artifact_paths)

    elapsed = int(time.time() - t0)
    results = []
    for cat in data_by_cat:
        m = cv_per_cat.get(cat, {})
        results.append({
            "experiment":         experiment,
            "mode":               "global",
            "category":           cat,
            **m,
            "beats_baseline":     (m.get("mape") or 999) < baseline_mapes.get(cat, 999),
            "forecast_skill_pct": skill_scores.get(cat, float("nan")),
            "is_overfit":         overfit_info.get("is_overfit", False),
            "best_epoch":         overfit_info.get("best_epoch", -1),
            "normalization":      "layer" if USE_LAYERNORM else "batch",
            "elapsed_sec":        elapsed,
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global DEBUG_FOLDS, USE_LAYERNORM

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true",
        help="Skip walk-forward CV — train final model only (~4x faster)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK,
        help=f"Lookback window in days (default: {LOOKBACK})")
    parser.add_argument("--tune-lookback", action="store_true",
        help="Compare --lookback-candidates and use whichever wins.")
    parser.add_argument("--lookback-candidates", type=str,
        default=",".join(str(x) for x in CANDIDATE_LOOKBACKS),
        help=f"Comma-separated lookback days to compare with --tune-lookback "
             f"(default: {','.join(str(x) for x in CANDIDATE_LOOKBACKS)})")
    parser.add_argument("--tune-hp", action="store_true",
        help="Run an Optuna hyperparameter search before training.")
    parser.add_argument("--n-trials", type=int, default=20,
        help="Number of Optuna trials for --tune-hp (default: 20)")
    parser.add_argument("--no-feature-importance", action="store_true",
        help="Skip permutation feature importance")
    parser.add_argument("--no-prediction-intervals", action="store_true",
        help="Skip residual-based prediction interval computation")
    parser.add_argument("--debug-folds", action="store_true",
        help="Print per-fold actual/predicted ranges (diagnostic, off by default)")
    parser.add_argument("--use-layernorm", action="store_true",
        help="Use LayerNormalization instead of BatchNormalization (experimental).")
    args = parser.parse_args()

    DEBUG_FOLDS   = args.debug_folds
    USE_LAYERNORM = args.use_layernorm

    _seed_tensorflow()

    lookback           = args.lookback
    compute_importance = not args.no_feature_importance
    compute_intervals  = not args.no_prediction_intervals

    log.info("=" * 55)
    log.info("LSTM DEMAND FORECASTING — Production Pipeline (anchor/delta head)")
    log.info(f"Lookback: {lookback}d | Forecast: {FORECAST_DAYS}d | Features: {N_FEATURES}")
    log.info(f"Embed dim: {EMBED_DIM} | Global seed: {SEED} | Log-transform: ON | "
             f"Output: DELTA-from-last-known-value")
    log.info(f"Normalization: {'LayerNorm' if USE_LAYERNORM else 'BatchNorm'}")
    log.info(f"Feature importance: {'ON' if compute_importance else 'OFF'} | "
             f"Prediction intervals: {'ON' if compute_intervals else 'OFF'} | "
             f"Debug folds: {'ON' if DEBUG_FOLDS else 'OFF'}")
    if args.skip_cv:
        log.info("Mode: FINAL TRAINING ONLY (--skip-cv)")
    log.info("=" * 55)

    df             = load_data()
    target_scaler, _ = load_scalers()
    baseline_mapes = load_baseline_mapes()

    experiment   = "smoothed"
    base_exp_df      = prepare_experiment_df(df)
    base_data_by_cat = prepare_data_by_category(base_exp_df, TOP_CATEGORIES, lookback=lookback)
    if not base_data_by_cat:
        log.error("No categories have enough history for the requested lookback — aborting")
        sys.exit(1)
    cat_to_id = build_category_map(list(base_data_by_cat.keys()))

    hp = DEFAULT_HP
    if args.tune_lookback:
        candidates = tuple(int(x) for x in args.lookback_candidates.split(","))
        log.info(f"\nComparing lookback candidates: {candidates}")
        lookback, _ = compare_lookbacks(base_data_by_cat, cat_to_id, target_scaler,
                                        candidates=candidates)
        base_exp_df      = prepare_experiment_df(df)
        base_data_by_cat = prepare_data_by_category(base_exp_df, TOP_CATEGORIES, lookback=lookback)
        cat_to_id = build_category_map(list(base_data_by_cat.keys()))

    if args.tune_hp:
        log.info(f"\nRunning Optuna hyperparameter search ({args.n_trials} trials)...")
        hp = run_optuna_search(base_data_by_cat, cat_to_id, target_scaler,
                               lookback=lookback, n_trials=args.n_trials)

    all_results = run_experiment_global(
        df, experiment, target_scaler, baseline_mapes,
        args.skip_cv, lookback, hp, compute_importance, compute_intervals)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(PROCESSED_DIR / "lstm_results.csv", index=False)

    with open(PROCESSED_DIR / "lstm_final_report.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info("\n" + "=" * 55)
    log.info("FINAL RESULTS")
    log.info("=" * 55)

    if results_df.empty:
        log.warning("No results produced — check warnings above for skipped categories")
        return

    show = [c for c in ["experiment", "category", "mape", "mase",
                        "forecast_skill_pct", "beats_baseline", "is_overfit"]
            if c in results_df.columns]
    log.info(f"\n{results_df[show].round(2).to_string(index=False)}")

    if "mape" in results_df.columns:
        agg_cols = [c for c in ["mape", "mase", "forecast_skill_pct"]
                   if c in results_df.columns]
        summary = results_df[agg_cols].mean().round(2)
        log.info(f"\nOverall averages:\n{summary.to_string()}")

        avg_mape  = results_df["mape"].mean()
        avg_skill = results_df["forecast_skill_pct"].mean() if "forecast_skill_pct" in results_df.columns else float("nan")
        n_beats   = int(results_df["beats_baseline"].sum()) if "beats_baseline" in results_df.columns else 0
        n_total   = len(results_df)

        log.info("\n── RESUME LINE ──")
        log.info(f"  LSTM demand forecasting (global model, smoothed): "
                 f"{avg_mape:.1f}% avg MAPE, beats baseline on {n_beats}/{n_total} categories")
        if not np.isnan(avg_skill):
            if avg_skill >= 0:
                log.info(f"  {avg_skill:.0f}% improvement over best baseline on 7-day horizon")
            else:
                log.info(f"  {abs(avg_skill):.0f}% WORSE than best baseline on 7-day horizon "
                         f"— do not report this as an improvement")

    log.info(f"\nModels  → {MODELS_DIR}")
    log.info(f"Results → {PROCESSED_DIR / 'lstm_results.csv'}")
    log.info("MLflow  → mlflow ui --port 5000")


if __name__ == "__main__":
    main()