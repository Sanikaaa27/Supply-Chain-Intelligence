"""LightGBM Demand Forecasting — Production Pipeline (no blending, pure model output)

CHANGE LOG v2.0 (this version) vs noblend version:
1. POOLED MASE BUG FIXED. Previous version's pooled MASE silently printed as
   nan because `all_metrics()` was called on the pooled actual/pred arrays
   WITHOUT a `train` argument anywhere in the pooled aggregation path — the
   key was simply missing from every fold's metrics dict, so nanmean over a
   missing key produced nan. Pooled `train_raw` is now concatenated across
   categories per fold and passed through, exactly like the per-category path.
2. QUANTILE CROSSING ENFORCED. Independent low/high quantile models can predict
   lo > hi for some rows, which silently corrupts interval width with a
   negative number. `lo`/`hi` are now sorted per-row with np.minimum/maximum
   before any width/coverage calculation.
3. CONFORMAL CALIBRATION ADDED. Raw quantile regression intervals were
   miscalibrated (80% interval -> 64.7% actual coverage, 95% -> 87.2%).
   Instead of trusting the raw quantile output, intervals are now widened
   post-hoc using empirical residual quantiles from the es_val slice
   (split-conformal style), and the corrected coverage is reported alongside
   the raw (pre-calibration) coverage so the improvement is visible, not just
   asserted.
4. NOISE THRESHOLD IS NOW DATA-DRIVEN. NOISE_THRESHOLD_PP was a hardcoded
   1.5pp guess. It's now derived per-run from the std of each category's MAPE
   across the 3 CV folds (averaged across categories) — if someone asks "why
   1.5pp," the old answer was "I picked it." The new answer is "it's
   approximately one fold-to-fold standard deviation of MAPE, the threshold
   below which a gap is indistinguishable from CV noise on this data."
5. PER-CATEGORY SHAP ADDED. Previous SHAP importance was computed on a
   pooled sample across all categories, which can hide a category whose
   importance profile differs sharply from the rest (this is exactly how
   telephony's failure went undiagnosed). Now also computes and logs
   per-category top-5 SHAP features for the worst-performing category by
   lift gap, so a `loses` category gets an actual root-cause artifact instead
   of just a number.
6. Hyperparameters for quantile models are now tunable separately from point
   models via --quantile-num-leaves / --quantile-learning-rate, since there
   is no reason to assume point-forecast hyperparameters are optimal for
   quantile regression objectives. Defaults still match point models for
   backward comparability, but they are no longer silently hardcoded as "the
   only option."

CHANGE LOG v2.1 (this version) vs v2.0:
7. DIRECTIONAL ACCURACY ADDED. MAPE/MASE only tell you how close the
   magnitude was — they say nothing about whether the model called the
   up/down move correctly, which is what an inventory "order more / order
   less" decision actually depends on. Added `directional_accuracy()` plus
   a genuine, non-strawman persistence baseline for direction: persistence
   predicts "no change," so it is scored correct only on days where the
   actual move is within DIRECTION_FLAT_EPS of flat, and wrong otherwise
   (it never predicts a direction, so it cannot get a real up/down move
   right). This makes the baseline beatable in principle, not a 0%-accuracy
   strawman. Computed per-category and pooled, in walk-forward CV, using the
   SAME folds/predictions already computed for MAPE — no extra model
   training required. Reported as its own honest tally alongside the
   existing MAPE-based lift classification (the two can disagree: a category
   can lose on MAPE magnitude but still call direction correctly more often
   than persistence, or vice versa — both numbers are kept separate and
   neither is allowed to silently stand in for the other).
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
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gbm")

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

FORECAST_DAYS  = 7
N_FOLDS        = 3
TEST_DAYS      = 30
MIN_TRAIN_DAYS = 120
ES_VAL_FRAC    = 0.10

# FIX 2.0 (#4): this is now a FALLBACK only, used if fold-level MAPE std
# can't be computed (e.g. --skip-cv runs with no fold history). When CV runs,
# the threshold is recomputed empirically — see `derive_noise_threshold()`.
DEFAULT_NOISE_THRESHOLD_PP = 1.5

PI_LEVELS = (0.8, 0.95)
QUANTILE_ALPHAS = {0.8: (0.10, 0.90), 0.95: (0.025, 0.975)}

USE_SHAP = True

# FIX 2.1 (#7): a day-over-day move smaller than this fraction of the anchor
# value counts as "flat" for direction-scoring purposes. This is what makes
# persistence a real, beatable baseline instead of a 0%-accuracy strawman —
# persistence is "correct" exactly on the days where demand barely moved,
# and "wrong" on every genuine up/down move (since it never predicts one).
DIRECTION_FLAT_EPS = 0.02  # 2% relative move treated as "no change"

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

DEFAULT_LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "mae",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l2":        1e-3,
    "min_child_samples": 15,
    "verbosity":        -1,
    "seed":             SEED,
}

# FIX 2.0 (#6): separate, independently tunable defaults for quantile models.
# Quantile loss surfaces behave differently from MAE/Huber point-forecast
# loss; reusing point-model hyperparameters unexamined was a real gap.
DEFAULT_QUANTILE_PARAMS = {
    "num_leaves":       31,
    "learning_rate":    0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l2":        1e-3,
    "min_child_samples": 20,   # slightly higher — quantile loss is noisier on small leaves
    "verbosity":        -1,
    "seed":             SEED,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
# FIX 2.0 (#6): quantile models get more patience — quantile loss converges
# less smoothly than MAE, and the old shared EARLY_STOPPING_ROUNDS=50 (tuned
# implicitly for the point models) was cutting these off early, which is part
# of why the raw intervals were too narrow.
QUANTILE_EARLY_STOPPING_ROUNDS = 80


# ── Metrics ──────────────────────────────────────────────────────────────

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
    """All three arrays must be in the SAME (raw) scale — caller's responsibility."""
    naive_errors = np.abs(np.diff(train))
    if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / np.mean(naive_errors))


def forecast_skill(model_mape, baseline_mape):
    if baseline_mape is None or baseline_mape == 0 or np.isnan(baseline_mape):
        return float("nan")
    return float((baseline_mape - model_mape) / baseline_mape * 100)


def classify_lift(model_mape, baseline_mape, noise_threshold):
    """Honest three-way label instead of a beats_baseline boolean.
    genuine_beat : model meaningfully better than baseline (gap > threshold)
    no_signal    : model and baseline within noise of each other
    loses        : model meaningfully worse than baseline
    """
    if baseline_mape is None or np.isnan(model_mape) or np.isnan(baseline_mape):
        return "unknown"
    gap = baseline_mape - model_mape   # positive = model better
    if gap > noise_threshold:
        return "genuine_beat"
    elif gap < -noise_threshold:
        return "loses"
    else:
        return "no_signal"


# FIX 2.1 (#7): direction-of-move scoring. Operates on (anchor, actual,
# predicted) raw-scale arrays. `anchor` is the last known value the forecast
# was made from (same anchor used elsewhere in this file for persistence).
def _sign_with_flat_band(delta: np.ndarray, ref: np.ndarray, eps: float = DIRECTION_FLAT_EPS) -> np.ndarray:
    """Returns -1/0/+1 per element: 0 if |delta| is within eps of ref (flat),
    else sign(delta). `ref` is the anchor value, used to make the flat-band
    relative rather than an absolute unit count (so it works across
    categories with very different demand scales)."""
    ref_safe = np.clip(np.abs(ref), 1e-6, None)
    rel_delta = delta / ref_safe
    out = np.sign(rel_delta)
    out[np.abs(rel_delta) < eps] = 0
    return out


def directional_accuracy(anchor: np.ndarray, actual: np.ndarray, predicted: np.ndarray,
                          eps: float = DIRECTION_FLAT_EPS) -> dict:
    """Compares model's predicted direction (vs anchor) to actual direction
    (vs anchor), per forecast step, and also scores a genuine persistence
    baseline (which always predicts 'flat' / no-change) on the SAME actual
    direction labels. anchor/actual/predicted must all be raw-scale and the
    same shape (n_samples, n_horizon_days) or flattenable to that.

    Returns model accuracy, persistence accuracy, and the pointwise lift,
    plus the flat-band share of actual moves (diagnostic: if almost
    everything is 'flat', directional accuracy is not a meaningful metric
    for this category and that should be visible, not hidden).
    """
    anchor_b   = np.broadcast_to(anchor.reshape(-1, 1), actual.shape) if actual.ndim == 2 and anchor.ndim == 1 else anchor

    actual_dir    = _sign_with_flat_band(actual - anchor_b, anchor_b, eps)
    predicted_dir = _sign_with_flat_band(predicted - anchor_b, anchor_b, eps)
    persistence_dir = np.zeros_like(actual_dir)  # persistence always predicts "no change"

    model_correct       = (predicted_dir == actual_dir)
    persistence_correct = (persistence_dir == actual_dir)

    n = actual_dir.size
    if n == 0:
        return {
            "model_directional_acc": float("nan"),
            "persistence_directional_acc": float("nan"),
            "random_guess_directional_acc": float("nan"),
            "directional_lift_pp": float("nan"),
            "directional_lift_vs_random_pp": float("nan"),
            "flat_share": float("nan"),
            "n_obs": 0,
        }

    model_acc       = float(model_correct.sum()) / n
    persistence_acc = float(persistence_correct.sum()) / n
    flat_share       = float((actual_dir == 0).sum()) / n

    # FIX 2.2: analytic random-guess baseline. A 3-way random guesser that
    # picks {down, flat, up} with equal 1/3 probability each, scored against
    # the SAME actual_dir labels, has expected accuracy:
    #   P(correct) = (1/3) * P(actual==flat) + (1/3) * P(actual==up) + (1/3) * P(actual==down)
    #              = 1/3   (always, regardless of the true label distribution)
    # That's the right comparison for an unbiased 3-way guesser, but
    # persistence is NOT a 3-way guesser — it always calls "flat", so the
    # fairer apples-to-apples random baseline is one that mimics how often a
    # naive observer could get "up" or "down" right purely by chance, i.e. a
    # 2-way coin flip restricted to the non-flat days, plus automatic credit
    # on flat days only if it happens to guess flat. We report the simple
    # unbiased 3-way figure (exactly 33.3%) since it requires no assumptions
    # about the guesser's flat-calling behavior and is the standard
    # "no-skill" reference point: if model accuracy isn't comfortably above
    # ~33%, beating a near-0% persistence baseline is not evidence of skill,
    # it's evidence persistence is a degenerate baseline for this eps.
    random_guess_acc = 100.0 / 3.0

    return {
        "model_directional_acc": model_acc * 100,
        "persistence_directional_acc": persistence_acc * 100,
        "random_guess_directional_acc": random_guess_acc,
        "directional_lift_pp": (model_acc - persistence_acc) * 100,
        "directional_lift_vs_random_pp": (model_acc * 100) - random_guess_acc,
        "flat_share": flat_share * 100,
        "n_obs": int(n),
    }


def classify_directional_lift(lift_pp: float, noise_threshold_pp: float) -> str:
    """Same honest three-way labeling logic as classify_lift(), reused for
    directional accuracy instead of MAPE. A model needs to beat persistence's
    directional accuracy by more than the noise threshold (in percentage
    points of accuracy) to count as a genuine win — NOT just any positive
    lift, which is exactly the shortcut that made the original 9/10 number
    meaningless."""
    if np.isnan(lift_pp):
        return "unknown"
    if lift_pp > noise_threshold_pp:
        return "genuine_beat"
    elif lift_pp < -noise_threshold_pp:
        return "loses"
    else:
        return "no_signal"


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
    else:
        # FIX 2.0 (#1): explicitly set nan instead of leaving the key absent.
        # An absent key silently breaks nanmean aggregation downstream in a
        # way that's hard to notice; an explicit nan at least shows up if you
        # print the dict, and np.nanmean handles it correctly either way.
        m["mase"] = float("nan")
    return m


def to_raw_scale(values: np.ndarray, target_scaler, clip_negative: bool = False) -> np.ndarray:
    flat = values.reshape(-1, 1)
    if clip_negative:
        flat = np.clip(flat, 0, None)
    inv = target_scaler.inverse_transform(flat).flatten()
    if clip_negative:
        inv = np.clip(inv, 0, None)
    return np.expm1(inv).reshape(values.shape)


def persistence_forecast(anchor_raw: np.ndarray, forecast_days: int = FORECAST_DAYS) -> np.ndarray:
    return np.repeat(anchor_raw.reshape(-1, 1), forecast_days, axis=1)


# ── Data loading ─────────────────────────────────────────────────────────

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


def prepare_experiment_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for cat in df["category"].unique():
        mask = df["category"] == cat
        df.loc[mask, "units_sold"] = (
            df.loc[mask, "units_sold"].rolling(7, min_periods=1).mean()
        )
    return df


def build_per_category_folds(data_by_cat: dict, n_folds: int = N_FOLDS,
                              test_days: int = TEST_DAYS,
                              min_train_days: int = MIN_TRAIN_DAYS) -> dict:
    per_cat_folds = {}
    for cat, data in data_by_cat.items():
        n = len(data)
        folds = []
        for i in range(n_folds):
            test_end   = n - (n_folds - i - 1) * test_days
            test_start = test_end - test_days
            train_end  = test_start
            if train_end < min_train_days:
                continue
            if test_start >= test_end or test_end > n:
                continue
            folds.append((train_end, test_start, test_end, i + 1))
        per_cat_folds[cat] = folds
    return per_cat_folds


def compute_smoothed_baseline_mapes(data_by_cat: dict, target_scaler,
                                     n_folds: int = N_FOLDS,
                                     test_days: int = TEST_DAYS,
                                     seasonal_period: int = 7) -> dict:
    per_cat_folds = build_per_category_folds(data_by_cat, n_folds=n_folds, test_days=test_days)
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
        out[cat] = {**method_means, "best_baseline": method_means[best_method],
                    "best_baseline_method": best_method}

    return out


def print_smoothed_baseline_winners(smoothed_baselines: dict) -> None:
    if not smoothed_baselines:
        log.warning("  No smoothed baselines computed — skipping printout")
        return
    log.info("  SMOOTHED-TARGET baseline winners:")
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


# ── Feature matrix construction ─────────────────────────────────────────

def build_feature_matrix(cat_df: pd.DataFrame, cat_name: str) -> pd.DataFrame:
    n = len(cat_df)
    rows = cat_df[FEATURE_COLS].values.astype(np.float64)

    usable_n = n - FORECAST_DAYS
    if usable_n <= 0:
        return pd.DataFrame()

    out = pd.DataFrame(rows[:usable_n], columns=FEATURE_COLS)
    out["category"] = cat_name
    out["category"] = out["category"].astype("category")
    out["anchor"]    = rows[:usable_n, 0]
    out["row_idx"]   = np.arange(usable_n)

    target_col = rows[:, 0]
    for d in range(1, FORECAST_DAYS + 1):
        out[f"target_d{d}"] = target_col[d: d + usable_n]

    return out


def build_all_feature_matrices(data_by_cat_df: dict) -> pd.DataFrame:
    parts = [build_feature_matrix(df, cat) for cat, df in data_by_cat_df.items()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    pooled = pd.concat(parts, ignore_index=True)
    pooled["category"] = pooled["category"].astype("category")
    return pooled


def prepare_data_by_category_df(df: pd.DataFrame, categories: list,
                                 min_rows: int = MIN_TRAIN_DAYS + FORECAST_DAYS) -> dict:
    out = {}
    for cat in categories:
        cat_df = (df[df["category"] == cat]
                  .sort_values("sale_date")
                  .reset_index(drop=True))
        n = len(cat_df)
        if n < min_rows:
            log.warning(f"  {cat}: {n} rows insufficient (need >= {min_rows}) — skipping")
            continue
        out[cat] = cat_df
    return out


# ── Per-horizon-day model training ──────────────────────────────────────

def train_one_horizon_model(train_df: pd.DataFrame, es_df: pd.DataFrame, day: int,
                             params: dict = None, quantile_alpha: float = None,
                             quantile_params: dict = None) -> lgb.Booster:
    """FIX 2.0 (#6): quantile models now use their own param dict and their
    own (longer) early stopping patience, set via QUANTILE_EARLY_STOPPING_ROUNDS,
    instead of silently inheriting point-model settings."""
    if quantile_alpha is not None:
        p = dict(quantile_params or DEFAULT_QUANTILE_PARAMS)
        p["objective"] = "quantile"
        p["alpha"] = quantile_alpha
        p["metric"] = "quantile"
        es_rounds = QUANTILE_EARLY_STOPPING_ROUNDS
    else:
        p = dict(params or DEFAULT_LGB_PARAMS)
        es_rounds = EARLY_STOPPING_ROUNDS

    feature_cols = FEATURE_COLS + ["category", "anchor"]
    target_col   = f"target_d{day}"

    train_set = lgb.Dataset(train_df[feature_cols], label=train_df[target_col],
                            categorical_feature=["category"], free_raw_data=False)
    es_set    = lgb.Dataset(es_df[feature_cols], label=es_df[target_col],
                            categorical_feature=["category"], reference=train_set,
                            free_raw_data=False)

    model = lgb.train(
        p, train_set, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[es_set], valid_names=["es_val"],
        callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                  lgb.log_evaluation(period=0)],
    )
    return model


def predict_horizon(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    feature_cols = FEATURE_COLS + ["category", "anchor"]
    return model.predict(df[feature_cols], num_iteration=model.best_iteration)


def chronological_split_per_category(pooled_df: pd.DataFrame, es_val_frac: float = ES_VAL_FRAC,
                                      holdout_end_by_cat: dict = None) -> dict:
    train_parts, es_parts = [], []

    for cat in pooled_df["category"].cat.categories:
        cat_rows = pooled_df[pooled_df["category"] == cat].sort_values("row_idx")
        if holdout_end_by_cat is not None:
            if cat not in holdout_end_by_cat:
                continue
            cat_rows = cat_rows[cat_rows["row_idx"] < holdout_end_by_cat[cat]]

        n = len(cat_rows)
        if n == 0:
            continue

        es_size = max(1, int(n * es_val_frac))
        if n - es_size < 5:
            log.warning(f"  {cat}: insufficient rows ({n}) for split — skipping")
            continue

        es_parts.append(cat_rows.iloc[-es_size:])
        train_parts.append(cat_rows.iloc[: -es_size])

    out = {
        "train": pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(),
        "es":    pd.concat(es_parts, ignore_index=True) if es_parts else pd.DataFrame(),
    }
    for key in out:
        if not out[key].empty:
            out[key]["category"] = out[key]["category"].astype(
                pd.CategoricalDtype(categories=pooled_df["category"].cat.categories))
    return out


# ── Walk-Forward CV (no blend) ───────────────────────────────────────────

def walk_forward_cv(data_by_cat_df: dict, target_scaler, params: dict = None,
                     print_diagnostics: bool = True) -> tuple:
    params = params or DEFAULT_LGB_PARAMS
    per_cat_folds = build_per_category_folds(
        {cat: df for cat, df in data_by_cat_df.items()})

    if not any(per_cat_folds.values()):
        log.warning("  No valid folds — check category lengths vs MIN_TRAIN_DAYS/TEST_DAYS")
        return {}, {}, {}, {}, {}, {}, {}

    pooled_full = build_all_feature_matrices(data_by_cat_df)
    if pooled_full.empty:
        log.warning("  No usable feature rows for CV")
        return {}, {}, {}, {}, {}, {}, {}

    per_cat_fold_metrics  = defaultdict(list)
    per_horizon_fold_mape = defaultdict(list)
    pooled_fold_metrics   = []
    # FIX 2.0 (#4): collect raw per-category MAPE per fold so we can derive
    # a data-driven noise threshold afterward (std of MAPE across folds).
    per_cat_fold_mape_raw = defaultdict(list)
    # FIX 2.1 (#7): collect per-category directional accuracy per fold, plus
    # pooled anchor/actual/predicted arrays for a pooled directional summary.
    per_cat_fold_diracc   = defaultdict(list)
    pooled_anchor_all, pooled_dir_actual_all, pooled_dir_pred_all = [], [], []

    for fold_num in range(1, N_FOLDS + 1):
        holdout_ends, test_windows = {}, {}
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

        split = chronological_split_per_category(pooled_full, es_val_frac=ES_VAL_FRAC,
                                                  holdout_end_by_cat=holdout_ends)
        train_df, es_df = split["train"], split["es"]
        if train_df.empty:
            log.warning(f"  Fold {fold_num}: empty training split — skipping")
            continue

        test_parts = []
        for cat, (train_end, test_start, test_end) in test_windows.items():
            cat_rows = pooled_full[pooled_full["category"] == cat]
            test_parts.append(cat_rows[(cat_rows["row_idx"] >= test_start) &
                                       (cat_rows["row_idx"] < test_end)])
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
        if test_df.empty:
            log.warning(f"  Fold {fold_num}: no test rows — skipping")
            continue

        horizon_models = {}
        for day in range(1, FORECAST_DAYS + 1):
            horizon_models[day] = train_one_horizon_model(train_df, es_df, day, params=params)

        fold_actual_all, fold_pred_all, fold_train_all = [], [], []
        horizon_actual = defaultdict(list)
        horizon_pred    = defaultdict(list)

        for cat, (train_end, test_start, test_end) in test_windows.items():
            cat_test = test_df[test_df["category"] == cat]
            if cat_test.empty:
                continue

            model_preds_scaled = np.column_stack([
                predict_horizon(horizon_models[d], cat_test) for d in range(1, FORECAST_DAYS + 1)
            ])
            actual_scaled = cat_test[[f"target_d{d}" for d in range(1, FORECAST_DAYS + 1)]].values

            actual_raw   = to_raw_scale(actual_scaled, target_scaler, clip_negative=False)
            predicted_raw = to_raw_scale(model_preds_scaled, target_scaler, clip_negative=True)
            # FIX 2.1 (#7): anchor in raw scale — same "anchor" column already
            # built into the feature matrix (the value the forecast starts from).
            anchor_raw = to_raw_scale(cat_test["anchor"].values, target_scaler, clip_negative=False)

            data_arr  = data_by_cat_df[cat][FEATURE_COLS].values.astype(np.float64)
            train_raw = to_raw_scale(data_arr[:train_end, 0], target_scaler, clip_negative=False)

            m = all_metrics(actual_raw.flatten(), predicted_raw.flatten(), train=train_raw)
            per_cat_fold_metrics[cat].append(m)
            per_cat_fold_mape_raw[cat].append(m["mape"])

            # FIX 2.1 (#7): directional accuracy for this category/fold.
            dir_metrics = directional_accuracy(anchor_raw, actual_raw, predicted_raw)
            per_cat_fold_diracc[cat].append(dir_metrics)

            pooled_anchor_all.append(anchor_raw)
            pooled_dir_actual_all.append(actual_raw)
            pooled_dir_pred_all.append(predicted_raw)

            fold_actual_all.append(actual_raw.flatten())
            fold_pred_all.append(predicted_raw.flatten())
            # FIX 2.0 (#1): keep the per-category train_raw so we can build a
            # pooled train_raw array for this fold and pass it to all_metrics
            # for the pooled MASE — this is what was missing before.
            fold_train_all.append(train_raw)

            for d in range(FORECAST_DAYS):
                horizon_actual[d + 1].append(actual_raw[:, d])
                horizon_pred[d + 1].append(predicted_raw[:, d])

        pooled_actual = np.concatenate(fold_actual_all)
        pooled_pred   = np.concatenate(fold_pred_all)
        # FIX 2.0 (#1): pooled train series for MASE. Concatenating raw demand
        # series across categories isn't perfectly principled (different
        # categories have different scales), but it's strictly better than
        # silently omitting the metric — and it matches what "pooled MAPE"
        # already does (pools across categories too). Documented here so
        # nobody mistakes it for a per-category MASE.
        pooled_train  = np.concatenate(fold_train_all)
        pooled_m      = all_metrics(pooled_actual, pooled_pred, train=pooled_train)
        pooled_fold_metrics.append(pooled_m)

        for d in range(1, FORECAST_DAYS + 1):
            day_actual = np.concatenate(horizon_actual[d])
            day_pred   = np.concatenate(horizon_pred[d])
            per_horizon_fold_mape[d].append(mape(day_actual, day_pred))

        log.info(f"  Fold {fold_num} ({len(test_windows)} categories, each on its OWN last "
                 f"{TEST_DAYS}d, NO BLEND — raw model output): "
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
    horizon_summary = {d: float(np.nanmean(v)) for d, v in per_horizon_fold_mape.items()}

    # FIX 2.1 (#7): per-category directional summary, averaged across folds
    # exactly like per_cat_summary above (mean of fold-level dicts).
    per_cat_diracc_summary = {
        cat: {k: float(np.nanmean([f[k] for f in fl])) for k in fl[0]}
        for cat, fl in per_cat_fold_diracc.items() if fl
    }

    # FIX 2.1 (#7): pooled directional summary, computed once over the full
    # concatenated pooled arrays (consistent with how pooled MAPE is done —
    # pooled over all test points across categories and folds, not an
    # average of per-category directional accuracies).
    pooled_diracc_summary = {}
    if pooled_dir_actual_all:
        pooled_anchor_arr = np.concatenate([a.flatten() if a.ndim > 1 else a for a in pooled_anchor_all]) \
            if pooled_anchor_all[0].ndim == 1 else np.concatenate(pooled_anchor_all)
        pooled_actual_dir_arr = np.concatenate(pooled_dir_actual_all)
        pooled_pred_dir_arr   = np.concatenate(pooled_dir_pred_all)
        # anchor arrays are 1-D per row (one anchor per forecast origin) while
        # actual/predicted are 2-D (n_rows, FORECAST_DAYS) — directional_accuracy
        # broadcasts a 1-D anchor against a 2-D actual/predicted correctly.
        pooled_anchor_1d = np.concatenate([a if a.ndim == 1 else a[:, 0] for a in pooled_anchor_all])
        pooled_diracc_summary = directional_accuracy(
            pooled_anchor_1d, pooled_actual_dir_arr, pooled_pred_dir_arr)

    if print_diagnostics and horizon_summary:
        log.info("  Pooled MAPE by forecast horizon (day 1 = tomorrow, day 7 = 1 week out):")
        for d in sorted(horizon_summary):
            bar = "█" * int(horizon_summary[d] / 2)
            log.info(f"    day {d}: {horizon_summary[d]:>5.1f}%  {bar}")

    return (per_cat_summary, pooled_summary, horizon_summary, per_cat_fold_mape_raw,
            per_cat_diracc_summary, pooled_diracc_summary, per_cat_fold_diracc)


def derive_noise_threshold(per_cat_fold_mape_raw: dict,
                            fallback: float = DEFAULT_NOISE_THRESHOLD_PP) -> float:
    """FIX 2.0 (#4): instead of a hardcoded 1.5pp guess, compute the average
    fold-to-fold MAPE standard deviation across categories. A gap vs baseline
    smaller than ~1 std of the model's own fold-to-fold variance isn't
    distinguishable from noise on this data — that's the actual justification
    for a noise threshold, now backed by the numbers instead of asserted."""
    stds = [float(np.std(v)) for v in per_cat_fold_mape_raw.values() if len(v) >= 2]
    if not stds:
        log.warning(f"  Could not derive noise threshold from fold variance — "
                    f"falling back to {fallback}pp")
        return fallback
    threshold = float(np.mean(stds))
    log.info(f"  Derived noise threshold: {threshold:.2f}pp "
             f"(mean fold-to-fold MAPE std across {len(stds)} categories, "
             f"vs hardcoded fallback of {fallback}pp)")
    return threshold


def derive_directional_noise_threshold(per_cat_fold_diracc: dict,
                                        fallback: float = 5.0) -> float:
    """FIX 2.1 (#7): same logic as derive_noise_threshold(), applied to
    directional accuracy instead of MAPE — the noise threshold here is the
    mean fold-to-fold standard deviation of the model's OWN directional
    accuracy (in percentage points), not an assumed number. Takes the raw
    per-fold directional accuracy dicts returned by walk_forward_cv
    (per_cat_fold_diracc), exactly mirroring how derive_noise_threshold()
    consumes per_cat_fold_mape_raw."""
    stds = [float(np.std([f["model_directional_acc"] for f in fl]))
            for fl in per_cat_fold_diracc.values() if len(fl) >= 2]
    if not stds:
        log.warning(f"  Could not derive directional noise threshold from fold variance — "
                    f"falling back to {fallback}pp")
        return fallback
    threshold = float(np.mean(stds))
    log.info(f"  Derived directional noise threshold: {threshold:.2f}pp "
             f"(mean fold-to-fold directional-accuracy std across {len(stds)} categories, "
             f"vs fallback of {fallback}pp)")
    return threshold


# ── Final Training (no blend) ────────────────────────────────────────────

def train_final_models(data_by_cat_df: dict, target_scaler, params: dict = None,
                        quantile_params: dict = None) -> dict:
    params = params or DEFAULT_LGB_PARAMS
    quantile_params = quantile_params or DEFAULT_QUANTILE_PARAMS
    pooled_full = build_all_feature_matrices(data_by_cat_df)
    if pooled_full.empty:
        log.error("  No usable feature rows for final training")
        return {}

    split = chronological_split_per_category(pooled_full, es_val_frac=ES_VAL_FRAC)
    train_df, es_df = split["train"], split["es"]
    if train_df.empty:
        log.error("  Empty training split for final model")
        return {}

    log.info(f"  Final split sizes — train={len(train_df)}  es_val={len(es_df)}")

    point_models = {}
    quantile_models = defaultdict(dict)
    best_iters = {}

    for day in range(1, FORECAST_DAYS + 1):
        model = train_one_horizon_model(train_df, es_df, day, params=params)
        point_models[day] = model
        best_iters[day] = model.best_iteration
        log.info(f"  day {day}: best_iteration={model.best_iteration}  "
                 f"best_score={model.best_score['es_val']['l1']:.4f}")

        for level, (lo_a, hi_a) in QUANTILE_ALPHAS.items():
            quantile_models[day][lo_a] = train_one_horizon_model(
                train_df, es_df, day, quantile_alpha=lo_a, quantile_params=quantile_params)
            quantile_models[day][hi_a] = train_one_horizon_model(
                train_df, es_df, day, quantile_alpha=hi_a, quantile_params=quantile_params)

    return {
        "point_models":    point_models,
        "quantile_models": quantile_models,
        "best_iterations": best_iters,
        "train_df": train_df, "es_df": es_df,
        "pooled_full": pooled_full,
    }


def compute_final_model_directional_accuracy(point_models: dict, es_df: pd.DataFrame,
                                              target_scaler) -> dict:
    """FIX 2.3: directional accuracy for the actual FINAL shipped point models,
    not just the CV-averaged estimate from walk_forward_cv. Mirrors how
    compute_prediction_intervals_quantile() evaluates the final quantile
    models on es_df instead of only reporting a CV figure.

    Caveat (kept explicit rather than silently glossed over): es_df is the
    early-stopping validation slice used to pick each model's best_iteration,
    not a fully untouched holdout — the same caveat already applies to the
    interval-calibration numbers above. Treat this as a final sanity check on
    the shipped model, with the walk-forward CV directional numbers (computed
    on genuinely unseen folds) as the primary, more trustworthy estimate.
    """
    feature_cols = FEATURE_COLS + ["category", "anchor"]
    anchor_raw = to_raw_scale(es_df["anchor"].values, target_scaler, clip_negative=False)

    actual_scaled = np.column_stack([
        es_df[f"target_d{d}"].values for d in range(1, FORECAST_DAYS + 1)
    ])
    predicted_scaled = np.column_stack([
        predict_horizon(point_models[d], es_df) for d in range(1, FORECAST_DAYS + 1)
    ])

    actual_raw    = to_raw_scale(actual_scaled, target_scaler, clip_negative=False)
    predicted_raw = to_raw_scale(predicted_scaled, target_scaler, clip_negative=True)

    overall = directional_accuracy(anchor_raw, actual_raw, predicted_raw)

    per_category = {}
    for cat in es_df["category"].cat.categories:
        mask = (es_df["category"] == cat).values
        if mask.sum() == 0:
            continue
        per_category[cat] = directional_accuracy(
            anchor_raw[mask], actual_raw[mask], predicted_raw[mask])

    return {"overall": overall, "per_category": per_category}


def compute_prediction_intervals_quantile(quantile_models: dict, es_df: pd.DataFrame,
                                           target_scaler) -> dict:
    """FIX 2.0 (#2, #3): quantile crossing is now enforced (lo<=hi per row
    before any width/coverage math), and a split-conformal correction is
    applied: the raw interval is widened by the empirical (1-coverage_target)
    quantile of |residual| beyond the raw bound, measured on this same
    es_val slice. Both raw and conformal-corrected coverage are reported so
    the fix is verifiable, not just claimed."""
    out = {}
    actual_scaled_by_day = {
        d: es_df[f"target_d{d}"].values for d in range(1, FORECAST_DAYS + 1)
    }
    for day in range(1, FORECAST_DAYS + 1):
        day_out = {}
        actual_raw = to_raw_scale(actual_scaled_by_day[day], target_scaler, clip_negative=False)

        for level, (lo_a, hi_a) in QUANTILE_ALPHAS.items():
            lo_pred_scaled = predict_horizon(quantile_models[day][lo_a], es_df)
            hi_pred_scaled = predict_horizon(quantile_models[day][hi_a], es_df)

            # FIX 2.0 (#2): enforce lo <= hi per row, both in scaled and raw space.
            lo_pred_scaled, hi_pred_scaled = (np.minimum(lo_pred_scaled, hi_pred_scaled),
                                              np.maximum(lo_pred_scaled, hi_pred_scaled))

            lo_raw = to_raw_scale(lo_pred_scaled, target_scaler, clip_negative=True)
            hi_raw = to_raw_scale(hi_pred_scaled, target_scaler, clip_negative=True)
            lo_raw, hi_raw = np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw)

            raw_coverage = float(np.mean((actual_raw >= lo_raw) & (actual_raw <= hi_raw)))

            # FIX 2.0 (#3): split-conformal widening. Compute how far outside
            # the raw interval the actual values fall, on this same es_val
            # slice, and pad both bounds by the (1 - target_coverage) quantile
            # of that overshoot. This is a calibration set re-used as its own
            # correction set (not a fresh held-out slice), so treat the
            # corrected number as an improved estimate, not a guarantee —
            # but it directly fixes the systematic undershoot instead of just
            # reporting it.
            below_miss = np.clip(lo_raw - actual_raw, 0, None)
            above_miss = np.clip(actual_raw - hi_raw, 0, None)
            miss = np.maximum(below_miss, above_miss)
            pad = float(np.quantile(miss, level)) if len(miss) else 0.0

            lo_raw_cal = lo_raw - pad
            hi_raw_cal = hi_raw + pad
            cal_coverage = float(np.mean((actual_raw >= lo_raw_cal) & (actual_raw <= hi_raw_cal)))

            avg_width_scaled = float(np.mean(hi_pred_scaled - lo_pred_scaled))
            avg_width_raw     = float(np.mean(hi_raw - lo_raw))
            avg_width_raw_cal = float(np.mean(hi_raw_cal - lo_raw_cal))

            day_out[str(int(level * 100))] = {
                "avg_width_scaled":      avg_width_scaled,
                "avg_width_raw":         avg_width_raw,
                "avg_width_raw_calibrated": avg_width_raw_cal,
                "raw_empirical_coverage":      raw_coverage,
                "calibrated_empirical_coverage": cal_coverage,
                "conformal_pad":         pad,
                "target_coverage":       level,
                "well_calibrated_raw":   abs(raw_coverage - level) < 0.07,
                "well_calibrated":       abs(cal_coverage - level) < 0.07,
            }
        out[str(day)] = day_out
    return out


def compute_shap_importance(point_models: dict, es_df: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
    """Pooled SHAP across all categories (kept for the overall feature ranking)."""
    import shap
    feature_cols = FEATURE_COLS + ["category", "anchor"]
    sample = es_df.sample(n=min(max_samples, len(es_df)), random_state=SEED) if len(es_df) > 0 else es_df

    sample_enc = sample[feature_cols].copy()
    sample_enc["category"] = sample_enc["category"].astype(
        pd.CategoricalDtype(categories=es_df["category"].cat.categories))

    all_importances = []
    for day, model in point_models.items():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_enc)
        mean_abs = np.abs(shap_values).mean(axis=0)
        all_importances.append(mean_abs)

    avg_importance = np.mean(all_importances, axis=0)
    out = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": avg_importance})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out


def compute_per_category_shap(point_models: dict, es_df: pd.DataFrame,
                               categories: list, max_samples: int = 200) -> dict:
    """FIX 2.0 (#5): per-category SHAP, so a category with an outlier
    importance profile (e.g. telephony) doesn't get averaged away by the
    pooled SHAP computation. Returns {category: DataFrame[feature, mean_abs_shap]}.
    Runs only on a handful of categories by default (the caller decides which)
    to keep runtime reasonable — SHAP over 7 horizon models x N categories
    adds up fast."""
    import shap
    feature_cols = FEATURE_COLS + ["category", "anchor"]
    out = {}

    for cat in categories:
        cat_rows = es_df[es_df["category"] == cat]
        if cat_rows.empty:
            continue
        sample = cat_rows.sample(n=min(max_samples, len(cat_rows)), random_state=SEED)
        sample_enc = sample[feature_cols].copy()
        sample_enc["category"] = sample_enc["category"].astype(
            pd.CategoricalDtype(categories=es_df["category"].cat.categories))

        all_importances = []
        for day, model in point_models.items():
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_enc)
            all_importances.append(np.abs(shap_values).mean(axis=0))

        avg_importance = np.mean(all_importances, axis=0)
        df_out = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": avg_importance})
        out[cat] = df_out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return out


def plot_shap_importance(importance_df: pd.DataFrame, suffix: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance_df.head(15).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="steelblue")
    ax.set_xlabel("Mean |SHAP value| (avg across 7 horizon-day models)")
    title_suffix = f" — {suffix}" if suffix else ""
    ax.set_title(f"SHAP Feature Importance — LightGBM{title_suffix}")
    plt.tight_layout()
    fname = f"shap_importance_gbm_smoothed{('_' + suffix) if suffix else ''}.png"
    path = PLOTS_DIR / fname
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def diagnose_worst_category(cv_per_cat: dict, baseline_mapes: dict, lift_labels: dict,
                            data_by_cat_df: dict, target_scaler) -> "str | None":
    """FIX 2.0 (#5): identifies the category with the worst lift gap (most
    negative skill) among 'loses' categories and prints a residual-over-time
    summary for it — actual diagnostic signal instead of just flagging the
    number. Returns the category name so the caller can request its SHAP."""
    losing = {cat: m for cat, m in cv_per_cat.items() if lift_labels.get(cat) == "loses"}
    if not losing:
        return None

    worst_cat = min(losing, key=lambda c: forecast_skill(losing[c].get("mape", float("nan")),
                                                          baseline_mapes.get(c)))
    m = losing[worst_cat]
    bias = m.get("bias", float("nan"))
    direction = "over-forecasting" if bias > 0 else "under-forecasting"

    log.warning(f"  ⚠ Worst-performing category: {worst_cat} "
               f"(MAPE={m.get('mape', float('nan')):.1f}% vs baseline "
               f"{baseline_mapes.get(worst_cat, float('nan')):.1f}%)")
    log.warning(f"    Direction: model is systematically {direction} "
               f"(mean bias={bias:+.2f} raw units)")

    cat_df = data_by_cat_df.get(worst_cat)
    if cat_df is not None and len(cat_df) > 0:
        raw_series = to_raw_scale(cat_df[FEATURE_COLS].values[:, 0].astype(np.float64),
                                   target_scaler, clip_negative=False)
        recent = raw_series[-60:]
        log.warning(f"    Last 60 days demand range: [{recent.min():.1f}, {recent.max():.1f}], "
                   f"mean={recent.mean():.1f}, std={recent.std():.1f} "
                   f"(coefficient of variation={recent.std() / max(recent.mean(), 1e-6):.2f} — "
                   f"high CV suggests low/erratic volume is part of the problem, not just model fit)")

    return worst_cat


# ── MLflow ────────────────────────────────────────────────────────────────

def log_mlflow(run_label: str, params: dict, metrics: dict,
               artifact_paths: list = None, mlflow_experiment: str = None):
    try:
        import mlflow
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(mlflow_experiment or "supply_chain_gbm_smoothed_noblend")

        with mlflow.start_run(run_name=f"gbm_noblend_{run_label}"):
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

        log.info(f"  MLflow → {uri} | gbm_noblend_{run_label}")
    except Exception as e:
        log.error(f"  MLflow FAILED: {type(e).__name__}: {e}")
        log.error("  Start with: mlflow ui --port 5000")


def flatten_per_category_metrics(per_cat_metrics: dict) -> dict:
    flat = {}
    for cat, m in per_cat_metrics.items():
        for k, v in m.items():
            flat[f"{cat}__{k}"] = v
    return flat


# ── Experiment Runner ────────────────────────────────────────────────────

def run_experiment(df: pd.DataFrame, target_scaler,
                   skip_cv: bool, compute_importance: bool, compute_intervals: bool,
                   params: dict = None, quantile_params: dict = None) -> list:
    log.info(f"\n{'='*55}")
    log.info("EXPERIMENT SMOOTHED — LIGHTGBM, NO BLEND (raw model output only) [v2.1]")
    log.info("Target: 7-day rolling mean (smoothed demand trend, log-transformed) | "
             "Categorical: native")
    log.info(f"{'='*55}")

    t0 = time.time()
    exp_df = prepare_experiment_df(df)
    data_by_cat_df = prepare_data_by_category_df(exp_df, TOP_CATEGORIES)

    log.info("  Category series lengths (days):")
    for cat, cdf in sorted(data_by_cat_df.items(), key=lambda kv: -len(kv[1])):
        log.info(f"    {cat:<25} {len(cdf)}")

    if not data_by_cat_df:
        log.error("  No categories had sufficient data — aborting")
        return []

    data_by_cat_arr = {cat: cdf[FEATURE_COLS].values.astype(np.float64)
                       for cat, cdf in data_by_cat_df.items()}

    smoothed_baselines = compute_smoothed_baseline_mapes(data_by_cat_arr, target_scaler)
    print_smoothed_baseline_winners(smoothed_baselines)
    baseline_mapes = {cat: v["best_baseline"] for cat, v in smoothed_baselines.items()}

    cv_per_cat, cv_pooled, cv_horizon = {}, {}, {}
    per_cat_fold_mape_raw = {}
    cv_per_cat_diracc, cv_pooled_diracc = {}, {}
    noise_threshold = DEFAULT_NOISE_THRESHOLD_PP
    directional_noise_threshold = 5.0
    if not skip_cv:
        log.info(f"  Walk-Forward CV ({N_FOLDS} folds, 7 horizon-day models per fold, "
                 f"each category tested on its OWN last {TEST_DAYS}d, NO BLEND):")
        (cv_per_cat, cv_pooled, cv_horizon, per_cat_fold_mape_raw,
         cv_per_cat_diracc, cv_pooled_diracc, per_cat_fold_diracc) = walk_forward_cv(
            data_by_cat_df, target_scaler, params=params, print_diagnostics=True)
        if cv_pooled.get("mape") is not None:
            log.info(f"  Pooled CV → MAPE={cv_pooled['mape']:.1f}%  "
                     f"MASE={cv_pooled.get('mase', float('nan')):.2f}  Bias={cv_pooled.get('bias', 0):+.2f}")

        # FIX 2.0 (#4): derive the noise threshold from this run's own fold variance.
        noise_threshold = derive_noise_threshold(per_cat_fold_mape_raw)

        # FIX 2.1 (#7): pooled directional accuracy headline, printed right
        # after the MAPE headline so the two metrics sit side by side.
        # FIX 2.2: random-guess comparison added — beating persistence means
        # nothing on its own if persistence is a degenerate baseline (low
        # flat_share). The real bar is random_guess_directional_acc (33.3%).
        if cv_pooled_diracc:
            log.info(f"  Pooled directional accuracy → Model={cv_pooled_diracc['model_directional_acc']:.1f}%  "
                     f"Persistence={cv_pooled_diracc['persistence_directional_acc']:.1f}%  "
                     f"Random-guess={cv_pooled_diracc['random_guess_directional_acc']:.1f}%  "
                     f"Lift-vs-persistence={cv_pooled_diracc['directional_lift_pp']:+.1f}pp  "
                     f"Lift-vs-random={cv_pooled_diracc['directional_lift_vs_random_pp']:+.1f}pp  "
                     f"(flat-band share of actual moves: {cv_pooled_diracc['flat_share']:.1f}%)")
            if cv_pooled_diracc['flat_share'] < 15.0:
                log.warning(f"  ⚠ flat-band share is only {cv_pooled_diracc['flat_share']:.1f}% — "
                           f"persistence is a WEAK/DEGENERATE baseline at this eps "
                           f"(DIRECTION_FLAT_EPS={DIRECTION_FLAT_EPS}). Beating persistence here is "
                           f"NOT strong evidence of skill on its own — check lift-vs-random instead.")

        if cv_per_cat:
            log.info("  Per-category CV: GBM (raw, unblended) vs each category's own best baseline,"
                     f" with honest signal classification (noise threshold={noise_threshold:.2f}pp):")
            lift_counts = defaultdict(int)
            lift_labels = {}
            for cat, m in sorted(cv_per_cat.items(), key=lambda kv: -kv[1].get("mape", 0)):
                base     = baseline_mapes.get(cat)
                base_str = f"{base:.1f}%" if base is not None else "n/a"
                label = classify_lift(m.get("mape", float("nan")), base, noise_threshold)
                lift_labels[cat] = label
                lift_counts[label] += 1
                mark = {"genuine_beat": "✓✓", "no_signal": "≈ ", "loses": "✗✗", "unknown": "??"}[label]
                log.info(f"    {mark} {cat:<25} GBM={m.get('mape', float('nan')):>6.1f}%   "
                         f"baseline={base_str:>8}   [{label}]")
            log.info(f"  Honest tally (MAPE): genuine_beat={lift_counts['genuine_beat']}  "
                     f"no_signal(noise)={lift_counts['no_signal']}  loses={lift_counts['loses']}")

            # FIX 2.1 (#7): equivalent honest tally for directional accuracy,
            # using its own data-driven noise threshold (NOT the MAPE one —
            # the two metrics have different units and different fold
            # variance, reusing the MAPE threshold would be arbitrary).
            # FIX 2.2: also classify vs the random-guess baseline (33.3%),
            # using the SAME noise threshold magnitude (it's a pp-of-accuracy
            # threshold either way) — a category can show genuine_beat vs
            # persistence while showing no_signal/loses vs random, and that
            # gap is exactly the signal that persistence was a weak baseline.
            dir_lift_labels = {}
            dir_vs_random_labels = {}
            if cv_per_cat_diracc:
                directional_noise_threshold = derive_directional_noise_threshold(per_cat_fold_diracc)
                dir_lift_counts = defaultdict(int)
                dir_vs_random_counts = defaultdict(int)
                log.info("  Per-category CV: directional accuracy vs persistence AND vs random-guess "
                         f"(noise threshold={directional_noise_threshold:.2f}pp, random-guess={100/3:.1f}%):")
                for cat, dm in sorted(cv_per_cat_diracc.items(),
                                       key=lambda kv: -kv[1]["directional_lift_vs_random_pp"]):
                    label = classify_directional_lift(dm["directional_lift_pp"], directional_noise_threshold)
                    label_vs_random = classify_directional_lift(
                        dm["directional_lift_vs_random_pp"], directional_noise_threshold)
                    dir_lift_labels[cat] = label
                    dir_vs_random_labels[cat] = label_vs_random
                    dir_lift_counts[label] += 1
                    dir_vs_random_counts[label_vs_random] += 1
                    mark        = {"genuine_beat": "✓✓", "no_signal": "≈ ", "loses": "✗✗", "unknown": "??"}[label]
                    mark_random = {"genuine_beat": "✓✓", "no_signal": "≈ ", "loses": "✗✗", "unknown": "??"}[label_vs_random]
                    log.info(f"    {cat:<25} Model={dm['model_directional_acc']:>5.1f}%   "
                             f"vs Persistence({dm['persistence_directional_acc']:>4.1f}%): {mark} {label:<13}  "
                             f"vs Random(33.3%): {mark_random} {label_vs_random:<13}  "
                             f"flat={dm['flat_share']:>4.1f}%")
                log.info(f"  Honest tally (Directional vs persistence): genuine_beat={dir_lift_counts['genuine_beat']}  "
                         f"no_signal(noise)={dir_lift_counts['no_signal']}  loses={dir_lift_counts['loses']}")
                log.info(f"  Honest tally (Directional vs RANDOM GUESS): genuine_beat={dir_vs_random_counts['genuine_beat']}  "
                         f"no_signal(noise)={dir_vs_random_counts['no_signal']}  loses={dir_vs_random_counts['loses']}")
                if dir_lift_counts['genuine_beat'] > dir_vs_random_counts['genuine_beat']:
                    log.warning(f"  ⚠ {dir_lift_counts['genuine_beat']} categories beat persistence but only "
                               f"{dir_vs_random_counts['genuine_beat']} beat random guessing — this gap means "
                               f"persistence is a weak baseline here, NOT that the model lacks skill. "
                               f"Report the vs-RANDOM tally as the credible claim.")

            # FIX 2.0 (#5): diagnose the worst loser instead of just reporting it.
            worst_cat = diagnose_worst_category(cv_per_cat, baseline_mapes, lift_labels,
                                                data_by_cat_df, target_scaler)
        else:
            lift_labels = {}
            dir_lift_labels = {}
            dir_vs_random_labels = {}
            worst_cat = None
    else:
        lift_labels = {}
        dir_lift_labels = {}
        dir_vs_random_labels = {}
        worst_cat = None

    log.info("  Final model training (7 point models + quantile models for intervals, no blend):")
    final = train_final_models(data_by_cat_df, target_scaler, params=params,
                               quantile_params=quantile_params)
    if not final:
        return []

    point_models     = final["point_models"]
    quantile_models   = final["quantile_models"]
    es_df            = final["es_df"]

    artifact_paths = []

    for day, model in point_models.items():
        model_path = MODELS_DIR / f"gbm_smoothed_day{day}.txt"
        model.save_model(str(model_path))
        artifact_paths.append(str(model_path))
    log.info(f"  Saved 7 horizon-day models → {MODELS_DIR}")

    # FIX 2.3: directional accuracy of the actual SHIPPED final model, on its
    # own es_val slice — closes the gap where only the CV-averaged estimate
    # was reported. CV remains the primary number (genuinely unseen folds);
    # this is a sanity check that the final model is consistent with it.
    final_dir = compute_final_model_directional_accuracy(point_models, es_df, target_scaler)
    final_dir_overall = final_dir["overall"]
    log.info(f"  FINAL MODEL directional accuracy (on es_val, not CV) → "
             f"Model={final_dir_overall['model_directional_acc']:.1f}%  "
             f"Persistence={final_dir_overall['persistence_directional_acc']:.1f}%  "
             f"Random-guess={final_dir_overall['random_guess_directional_acc']:.1f}%  "
             f"Lift-vs-random={final_dir_overall['directional_lift_vs_random_pp']:+.1f}pp")
    if cv_pooled_diracc and not np.isnan(cv_pooled_diracc.get("model_directional_acc", float("nan"))):
        cv_vs_final_gap = abs(final_dir_overall["model_directional_acc"]
                               - cv_pooled_diracc["model_directional_acc"])
        if cv_vs_final_gap > 10.0:
            log.warning(f"  ⚠ Final model directional accuracy ({final_dir_overall['model_directional_acc']:.1f}%) "
                       f"diverges from CV estimate ({cv_pooled_diracc['model_directional_acc']:.1f}%) by "
                       f"{cv_vs_final_gap:.1f}pp — es_val is a small, non-independent slice, treat the CV "
                       f"number as primary and this as a rough consistency check only.")

    final_dir_path = PROCESSED_DIR / "gbm_final_model_directional_accuracy.json"
    with open(final_dir_path, "w") as f:
        json.dump(final_dir, f, indent=2, default=str)
    artifact_paths.append(str(final_dir_path))

    if compute_intervals:
        pi = compute_prediction_intervals_quantile(quantile_models, es_df, target_scaler)
        pi_path = PROCESSED_DIR / "gbm_smoothed_intervals.json"
        with open(pi_path, "w") as f:
            json.dump(pi, f, indent=2)
        artifact_paths.append(str(pi_path))
        for level in PI_LEVELS:
            lvl = str(int(level * 100))
            avg_width_raw     = float(np.mean([pi[str(d)][lvl]["avg_width_raw"]
                                               for d in range(1, FORECAST_DAYS + 1)]))
            avg_width_cal     = float(np.mean([pi[str(d)][lvl]["avg_width_raw_calibrated"]
                                               for d in range(1, FORECAST_DAYS + 1)]))
            avg_cov_raw       = float(np.mean([pi[str(d)][lvl]["raw_empirical_coverage"]
                                               for d in range(1, FORECAST_DAYS + 1)]))
            avg_cov_cal       = float(np.mean([pi[str(d)][lvl]["calibrated_empirical_coverage"]
                                               for d in range(1, FORECAST_DAYS + 1)]))
            calib_flag_raw = "well-calibrated" if abs(avg_cov_raw - level) < 0.07 else "MISCALIBRATED"
            calib_flag_cal = "well-calibrated" if abs(avg_cov_cal - level) < 0.07 else "MISCALIBRATED"
            log.info(f"  {lvl}% interval RAW        — width {avg_width_raw:.2f} (raw units)  |  "
                     f"coverage {avg_cov_raw*100:.1f}% (target {int(level*100)}%)  [{calib_flag_raw}]")
            log.info(f"  {lvl}% interval CALIBRATED — width {avg_width_cal:.2f} (raw units)  |  "
                     f"coverage {avg_cov_cal*100:.1f}% (target {int(level*100)}%)  [{calib_flag_cal}]")
        for day in range(1, FORECAST_DAYS + 1):
            for alpha, model in quantile_models[day].items():
                qpath = MODELS_DIR / f"gbm_smoothed_day{day}_q{alpha}.txt"
                model.save_model(str(qpath))

    if compute_importance:
        log.info("  Computing pooled SHAP feature importance (per horizon-day model, averaged)...")
        importance_df = compute_shap_importance(point_models, es_df)
        imp_path = PROCESSED_DIR / "shap_importance_gbm_smoothed.csv"
        importance_df.to_csv(imp_path, index=False)
        plot_shap_importance(importance_df)
        artifact_paths.append(str(imp_path))
        top5 = ", ".join(importance_df.head(5)["feature"].tolist())
        log.info(f"  Top 5 features by mean |SHAP| (pooled): {top5}")

        # FIX 2.0 (#5): per-category SHAP for the worst loser, so the failure
        # has an actual root-cause artifact, not just a flagged number.
        if worst_cat is not None:
            log.info(f"  Computing per-category SHAP for worst performer: {worst_cat}...")
            per_cat_shap = compute_per_category_shap(point_models, es_df, [worst_cat])
            if worst_cat in per_cat_shap:
                cat_imp_path = PROCESSED_DIR / f"shap_importance_gbm_smoothed_{worst_cat}.csv"
                per_cat_shap[worst_cat].to_csv(cat_imp_path, index=False)
                plot_shap_importance(per_cat_shap[worst_cat], suffix=worst_cat)
                artifact_paths.append(str(cat_imp_path))
                cat_top5 = ", ".join(per_cat_shap[worst_cat].head(5)["feature"].tolist())
                log.info(f"  Top 5 features for {worst_cat} (may differ from pooled ranking): {cat_top5}")

    skill_scores = {
        cat: forecast_skill(m.get("mape", float("nan")), baseline_mapes.get(cat))
        for cat, m in cv_per_cat.items()
    }
    valid_skills = [v for v in skill_scores.values() if not np.isnan(v)]
    avg_skill    = float(np.mean(valid_skills)) if valid_skills else float("nan")

    mlflow_params = {
        "model_family": "lightgbm", "multi_step_strategy": "per_horizon_day",
        "n_horizon_models": FORECAST_DAYS, "categorical_handling": "native",
        "log_transform": True, "blend_enabled": False,
        "noise_threshold_pp": noise_threshold,
        "directional_noise_threshold_pp": directional_noise_threshold,
        "direction_flat_eps": DIRECTION_FLAT_EPS,
        **{f"lgb_{k}": v for k, v in (params or DEFAULT_LGB_PARAMS).items()},
    }
    mlflow_metrics = {
        **{f"pooled_{k}": v for k, v in cv_pooled.items()},
        **flatten_per_category_metrics(cv_per_cat),
        **{f"skill_{cat}": v for cat, v in skill_scores.items() if not np.isnan(v)},
        **{f"horizon_day{d}_mape": v for d, v in cv_horizon.items()},
        "avg_forecast_skill_pct": avg_skill,
        "n_genuine_beat": sum(1 for v in lift_labels.values() if v == "genuine_beat"),
        "n_no_signal": sum(1 for v in lift_labels.values() if v == "no_signal"),
        "n_loses": sum(1 for v in lift_labels.values() if v == "loses"),
        # FIX 2.1 (#7): directional accuracy metrics logged alongside MAPE ones.
        **{f"pooled_dir_{k}": v for k, v in cv_pooled_diracc.items()},
        **flatten_per_category_metrics(cv_per_cat_diracc),
        "n_dir_genuine_beat": sum(1 for v in dir_lift_labels.values() if v == "genuine_beat"),
        "n_dir_no_signal": sum(1 for v in dir_lift_labels.values() if v == "no_signal"),
        "n_dir_loses": sum(1 for v in dir_lift_labels.values() if v == "loses"),
        # FIX 2.2: vs-random-guess tally — the credible directional claim.
        "n_dir_vs_random_genuine_beat": sum(1 for v in dir_vs_random_labels.values() if v == "genuine_beat"),
        "n_dir_vs_random_no_signal": sum(1 for v in dir_vs_random_labels.values() if v == "no_signal"),
        "n_dir_vs_random_loses": sum(1 for v in dir_vs_random_labels.values() if v == "loses"),
        # FIX 2.3: final SHIPPED model's own directional accuracy (on es_val),
        # kept separate from the pooled_dir_ (CV-averaged) fields above so
        # neither number silently overwrites or is confused with the other.
        **{f"final_model_dir_{k}": v for k, v in final_dir_overall.items()},
    }
    log_mlflow("smoothed", mlflow_params, mlflow_metrics, artifact_paths)

    elapsed = int(time.time() - t0)
    results = []
    for cat in data_by_cat_df:
        m = cv_per_cat.get(cat, {})
        dm = cv_per_cat_diracc.get(cat, {})
        results.append({
            "experiment":         "smoothed_noblend_v2",
            "model_family":       "lightgbm",
            "category":           cat,
            **m,
            "baseline_mape":      baseline_mapes.get(cat, float("nan")),
            "lift_label":         lift_labels.get(cat, "unknown"),
            "forecast_skill_pct": skill_scores.get(cat, float("nan")),
            "noise_threshold_pp": noise_threshold,
            # FIX 2.1 (#7): directional columns added to the per-category results row.
            "model_directional_acc":       dm.get("model_directional_acc", float("nan")),
            "persistence_directional_acc": dm.get("persistence_directional_acc", float("nan")),
            "directional_lift_pp":         dm.get("directional_lift_pp", float("nan")),
            "directional_flat_share_pct":  dm.get("flat_share", float("nan")),
            "directional_lift_label":      dir_lift_labels.get(cat, "unknown"),
            "directional_noise_threshold_pp": directional_noise_threshold,
            # FIX 2.2: vs-random-guess columns — the credible directional claim.
            "random_guess_directional_acc":   dm.get("random_guess_directional_acc", float("nan")),
            "directional_lift_vs_random_pp":  dm.get("directional_lift_vs_random_pp", float("nan")),
            "directional_lift_vs_random_label": dir_vs_random_labels.get(cat, "unknown"),
            "elapsed_sec":        elapsed,
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global USE_SHAP, DIRECTION_FLAT_EPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true",
        help="Skip walk-forward CV — train final model(s) only")
    parser.add_argument("--no-shap", action="store_true",
        help="Skip SHAP feature importance (faster)")
    parser.add_argument("--no-prediction-intervals", action="store_true",
        help="Skip quantile-regression prediction interval models")
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_LGB_PARAMS["num_leaves"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LGB_PARAMS["learning_rate"])
    # FIX 2.0 (#6): separately tunable quantile model hyperparameters.
    parser.add_argument("--quantile-num-leaves", type=int,
        default=DEFAULT_QUANTILE_PARAMS["num_leaves"])
    parser.add_argument("--quantile-learning-rate", type=float,
        default=DEFAULT_QUANTILE_PARAMS["learning_rate"])
    # FIX 2.1 (#7): the flat-band epsilon is exposed as a CLI flag so it can
    # be tuned/inspected without editing source — it directly controls how
    # strict "direction" means and is exactly the kind of knob that should
    # be visible, not buried.
    parser.add_argument("--direction-flat-eps", type=float, default=DIRECTION_FLAT_EPS,
        help="Relative move (fraction of anchor) treated as 'no change' for "
             "directional accuracy scoring (default: %(default)s)")
    args = parser.parse_args()

    USE_SHAP  = not args.no_shap
    compute_importance = USE_SHAP
    compute_intervals  = not args.no_prediction_intervals
    DIRECTION_FLAT_EPS = args.direction_flat_eps

    params = dict(DEFAULT_LGB_PARAMS)
    params["num_leaves"]    = args.num_leaves
    params["learning_rate"] = args.learning_rate

    quantile_params = dict(DEFAULT_QUANTILE_PARAMS)
    quantile_params["num_leaves"]    = args.quantile_num_leaves
    quantile_params["learning_rate"] = args.quantile_learning_rate

    log.info("=" * 55)
    log.info("LIGHTGBM DEMAND FORECASTING — Production Pipeline (NO BLEND) [v2.1]")
    log.info(f"Multi-step: {FORECAST_DAYS} independent per-horizon-day models | "
             f"Categorical: native | Log-transform: ON")
    log.info("Baseline blending: REMOVED — raw model output is the final forecast")
    log.info("v2.0 fixes: pooled MASE, quantile crossing, conformal interval "
             "calibration, data-driven noise threshold, per-category SHAP")
    log.info("v2.1 fix: directional accuracy (model vs honest persistence baseline), "
             f"flat-band eps={DIRECTION_FLAT_EPS}")
    log.info(f"SHAP importance: {'ON' if compute_importance else 'OFF'} | "
             f"Quantile prediction intervals: {'ON' if compute_intervals else 'OFF'}")
    if args.skip_cv:
        log.info("Mode: FINAL TRAINING ONLY (--skip-cv)")
    log.info("=" * 55)

    df               = load_data()
    target_scaler, _ = load_scalers()

    all_results = run_experiment(df, target_scaler,
                                 args.skip_cv, compute_importance, compute_intervals,
                                 params=params, quantile_params=quantile_params)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(PROCESSED_DIR / "gbm_results_noblend_v2.csv", index=False)

    with open(PROCESSED_DIR / "gbm_final_report_noblend_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info("\n" + "=" * 55)
    log.info("FINAL RESULTS (NO BLEND, v2.1 — raw model honesty check)")
    log.info("=" * 55)

    if results_df.empty:
        log.warning("No results produced — check warnings above")
        return

    show = [c for c in ["experiment", "category", "mape", "mase", "baseline_mape",
                        "lift_label", "forecast_skill_pct",
                        "model_directional_acc", "persistence_directional_acc",
                        "directional_lift_pp", "directional_lift_label"]
            if c in results_df.columns]
    log.info(f"\n{results_df[show].round(2).to_string(index=False)}")

    if "mape" in results_df.columns:
        agg_cols = [c for c in ["mape", "mase", "forecast_skill_pct",
                                "model_directional_acc", "persistence_directional_acc",
                                "directional_lift_pp"] if c in results_df.columns]
        summary = results_df[agg_cols].mean().round(2)
        log.info(f"\nOverall averages:\n{summary.to_string()}")

        avg_mape  = results_df["mape"].mean()
        n_genuine = int((results_df["lift_label"] == "genuine_beat").sum()) if "lift_label" in results_df.columns else 0
        n_noise   = int((results_df["lift_label"] == "no_signal").sum()) if "lift_label" in results_df.columns else 0
        n_loses   = int((results_df["lift_label"] == "loses").sum()) if "lift_label" in results_df.columns else 0
        n_total   = len(results_df)
        nthresh   = results_df["noise_threshold_pp"].iloc[0] if "noise_threshold_pp" in results_df.columns else DEFAULT_NOISE_THRESHOLD_PP

        log.info("\n── HONEST RESUME LINE (MAPE) ──")
        log.info(f"  LightGBM demand forecasting (raw model, no blending): {avg_mape:.1f}% avg MAPE. "
                 f"Of {n_total} categories (noise threshold={nthresh:.2f}pp, derived from CV fold "
                 f"variance): {n_genuine} show genuine model lift over persistence, "
                 f"{n_noise} are statistically indistinguishable from persistence, "
                 f"{n_loses} underperform persistence.")
        log.info("  Do NOT report 'beats baseline on X/10' without this breakdown —"
                 " it overstates the result.")

        if "directional_lift_label" in results_df.columns:
            n_dir_genuine = int((results_df["directional_lift_label"] == "genuine_beat").sum())
            n_dir_noise   = int((results_df["directional_lift_label"] == "no_signal").sum())
            n_dir_loses   = int((results_df["directional_lift_label"] == "loses").sum())
            avg_dir_lift  = results_df["directional_lift_pp"].mean()
            dthresh = results_df["directional_noise_threshold_pp"].iloc[0] \
                if "directional_noise_threshold_pp" in results_df.columns else float("nan")
            avg_flat_share = results_df["directional_flat_share_pct"].mean() \
                if "directional_flat_share_pct" in results_df.columns else float("nan")

            log.info("\n── DIRECTIONAL ACCURACY vs PERSISTENCE (secondary — check vs-random below) ──")
            log.info(f"  avg lift {avg_dir_lift:+.1f}pp vs persistence. Of {n_total} categories "
                     f"(noise threshold={dthresh:.2f}pp): {n_dir_genuine} genuine_beat, "
                     f"{n_dir_noise} no_signal, {n_dir_loses} loses.")
            if not np.isnan(avg_flat_share) and avg_flat_share < 15.0:
                log.warning(f"  ⚠ avg flat-band share is only {avg_flat_share:.1f}% — persistence rarely "
                           f"predicts correctly at this eps, making it an easy target. This tally alone "
                           f"OVERSTATES the result. Use the vs-RANDOM tally below as the real claim.")

        if "directional_lift_vs_random_label" in results_df.columns:
            n_vr_genuine = int((results_df["directional_lift_vs_random_label"] == "genuine_beat").sum())
            n_vr_noise   = int((results_df["directional_lift_vs_random_label"] == "no_signal").sum())
            n_vr_loses   = int((results_df["directional_lift_vs_random_label"] == "loses").sum())
            avg_vr_lift  = results_df["directional_lift_vs_random_pp"].mean()

            log.info("\n── HONEST RESUME LINE (DIRECTIONAL ACCURACY vs RANDOM GUESS) ──")
            log.info(f"  Directional accuracy (up/down call vs a no-skill 33.3% random-guess baseline): "
                     f"avg lift {avg_vr_lift:+.1f}pp. "
                     f"Of {n_total} categories: {n_vr_genuine} show genuine skill above chance, "
                     f"{n_vr_noise} are statistically indistinguishable from random guessing, "
                     f"{n_vr_loses} call direction worse than chance.")
            log.info("  This is the credible directional claim — the vs-persistence number above can "
                     "look inflated when persistence itself is a weak baseline (low flat-band share). "
                     "Report vs-random, not vs-persistence, as the headline.")
            log.info("  Both directional metrics remain SEPARATE from MAPE lift — a category can win on "
                     "one and not the other. Do not collapse any of these into a single claim.")

    log.info(f"\nModels  → {MODELS_DIR}")
    log.info(f"Results → {PROCESSED_DIR / 'gbm_results_noblend_v2.csv'}")
    log.info("MLflow  → mlflow ui --port 5000")


if __name__ == "__main__":
    main()