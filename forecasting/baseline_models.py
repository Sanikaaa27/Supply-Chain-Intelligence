"""
Baseline Models: Last-Value + Seasonal Naive + Average Demand + Prophet
Walk-forward validation | Coverage metric | MASE | Residual analysis

Run:    python forecasting/baseline_models.py
Prereq: feature_engineering.py must complete first

Outputs:
  data/processed/baseline_results.csv             — per-category metrics for all models
  data/processed/baseline_summary.json            — LSTM targets per category
  data/processed/prophet_forecasts.csv            — forecasts + confidence intervals
  data/plots/{model}/{category}/residuals.png     — residual distribution plots (per model/category)
  data/plots/model_comparison/avg_mape_by_model.png — cross-model MAPE comparison
  data/plots/model_comparison/avg_mase_by_model.png — cross-model MASE comparison

Baselines in order of complexity:
  1. Last-Value    — last observed value carried forward
  2. Average Demand — mean of last 30 days carried forward
  3. Seasonal Naive — same weekday 7 days ago (handles weekly cycles)
  4. Prophet        — additive seasonality, 90% confidence intervals

LSTM must beat best baseline by 15% MAPE to justify complexity.
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("baseline")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PLOTS_DIR     = Path(__file__).parent.parent / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS        = 3
TEST_DAYS      = 30
MIN_TRAIN_DAYS = 120


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    if mask.sum() < 5:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask  = denom > 0
    if mask.sum() < 5:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))

def mase(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error.
    Scales MAE by the in-sample naive forecast error (lag-1).
    MASE < 1 → better than naive. MASE > 1 → worse than naive.
    Preferred over MAPE for intermittent/zero-demand data.
    """
    naive_errors = np.abs(np.diff(train))
    if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / np.mean(naive_errors))

def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Interval coverage: % of actual values inside [lower, upper].
    For 90% intervals, should be ~90%. Lower = overconfident. Higher = too wide.
    Only meaningful for Prophet (produces intervals). Naive has no intervals.
    """
    inside = np.sum((actual >= lower) & (actual <= upper))
    return float(inside / len(actual) * 100)

def all_metrics(actual: np.ndarray, predicted: np.ndarray,
                train: np.ndarray,
                lower: np.ndarray = None,
                upper: np.ndarray = None) -> dict:
    m = {
        "mape":  mape(actual, predicted),
        "smape": smape(actual, predicted),
        "rmse":  rmse(actual, predicted),
        "mae":   mae(actual, predicted),
        "mase":  mase(actual, predicted, train),
        "bias":  float(np.mean(predicted - actual)),
    }
    if lower is not None and upper is not None:
        m["coverage_90"] = coverage(actual, lower, upper)
    return m


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, dict]:
    for p in [PROCESSED_DIR / "lstm_features.parquet",
              PROCESSED_DIR / "feature_dates.parquet"]:
        if not p.exists():
            log.error(f"{p.name} missing — run feature_engineering.py first")
            sys.exit(1)

    feat_df  = pd.read_parquet(PROCESSED_DIR / "lstm_features.parquet")
    dates_df = pd.read_parquet(PROCESSED_DIR / "feature_dates.parquet")

    # Inverse-transform units_sold → original scale for evaluation
    target_scaler             = joblib.load(PROCESSED_DIR / "target_scaler.pkl")
    feat_df["units_sold_raw"] = target_scaler.inverse_transform(
        feat_df[["units_sold"]]
    ).flatten()

    # Align dates back — sale_date excluded from feature parquet intentionally
    feat_df["sale_date"] = None
    for cat in feat_df["category"].unique():
        cat_mask  = feat_df["category"] == cat
        cat_dates = dates_df[dates_df["category"] == cat]["sale_date"].values
        n         = cat_mask.sum()
        feat_df.loc[cat_mask, "sale_date"] = cat_dates[:n]

    feat_df["sale_date"] = pd.to_datetime(feat_df["sale_date"])

    cat_stats = {}
    stats_path = PROCESSED_DIR / "category_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            cat_stats = json.load(f)

    log.info(f"Loaded {len(feat_df):,} rows | {feat_df['category'].nunique()} categories")
    return feat_df, cat_stats


# ── Fold Builder ──────────────────────────────────────────────────────────────

def build_folds(n: int) -> list[dict]:
    """
    Walk-forward folds with non-overlapping 30-day test windows.
    N_FOLDS=3 chosen deliberately — 4 folds caused empty train windows
    for low-volume categories in earlier testing.
    """
    folds = []
    for i in range(N_FOLDS):
        test_end   = n - (N_FOLDS - i - 1) * TEST_DAYS
        test_start = test_end - TEST_DAYS
        train_end  = test_start

        if train_end < MIN_TRAIN_DAYS:
            continue
        if test_start >= test_end or test_end > n:
            continue

        folds.append({
            "fold":       i + 1,
            "train":      (0, train_end),
            "test":       (test_start, test_end),
            "train_days": train_end,
            "test_days":  test_end - test_start,
        })
    return folds


# ── Baseline 1: Last-Value ────────────────────────────────────────────────────

def run_last_value(series: pd.Series, dates: pd.Series) -> tuple[dict, list[dict]]:
    """
    Last observation carried forward.
    Simplest possible baseline — if LSTM can't beat this, something is wrong.
    """
    folds          = build_folds(len(series))
    fold_metrics   = []
    fold_residuals = []

    for fold in folds:
        tr_s, tr_e = fold["train"]
        te_s, te_e = fold["test"]

        train     = series.iloc[tr_s:tr_e].values
        test      = series.iloc[te_s:te_e].values
        predicted = np.full_like(test, float(train[-1]), dtype=float)

        m = all_metrics(test, predicted, train)
        fold_metrics.append(m)

        for j in range(len(test)):
            fold_residuals.append({
                "date":      dates.iloc[te_s + j],
                "actual":    test[j],
                "predicted": predicted[j],
                "residual":  test[j] - predicted[j],
                "fold":      fold["fold"],
            })

        log.info(f"    Fold {fold['fold']} "
                 f"(train={fold['train_days']}d): "
                 f"MAPE={m['mape']:.1f}%  MASE={m['mase']:.2f}  Bias={m['bias']:+.2f}")

    if not fold_metrics:
        return {"model": "Last_Value", "mape": None, "n_folds": 0}, []

    avg = {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    avg["model"]   = "Last_Value"
    avg["n_folds"] = len(fold_metrics)
    return avg, fold_residuals


# ── Baseline 2: Average Demand ────────────────────────────────────────────────

def run_average_demand(series: pd.Series, dates: pd.Series,
                       window: int = 30) -> tuple[dict, list[dict]]:
    """
    Mean of last `window` days carried forward.
    Sometimes beats fancy models — important to test.
    If Average Demand beats Prophet, LSTM target needs recalibration.
    """
    folds          = build_folds(len(series))
    fold_metrics   = []
    fold_residuals = []

    for fold in folds:
        tr_s, tr_e = fold["train"]
        te_s, te_e = fold["test"]

        train     = series.iloc[tr_s:tr_e].values
        test      = series.iloc[te_s:te_e].values
        avg_val   = float(np.mean(train[-window:]))
        predicted = np.full_like(test, avg_val, dtype=float)

        m = all_metrics(test, predicted, train)
        fold_metrics.append(m)

        for j in range(len(test)):
            fold_residuals.append({
                "date":      dates.iloc[te_s + j],
                "actual":    test[j],
                "predicted": predicted[j],
                "residual":  test[j] - predicted[j],
                "fold":      fold["fold"],
            })

        log.info(f"    Fold {fold['fold']} "
                 f"(train={fold['train_days']}d, window={window}d): "
                 f"MAPE={m['mape']:.1f}%  MASE={m['mase']:.2f}  Bias={m['bias']:+.2f}")

    if not fold_metrics:
        return {"model": "Average_Demand", "mape": None, "n_folds": 0}, []

    avg = {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    avg["model"]   = "Average_Demand"
    avg["n_folds"] = len(fold_metrics)
    return avg, fold_residuals


# ── Baseline 3: Seasonal Naive ────────────────────────────────────────────────

def run_seasonal_naive(series: pd.Series, dates: pd.Series) -> tuple[dict, list[dict]]:
    """
    Seasonal naive: forecast = demand from same weekday 7 days ago.
    Outperforms last-value on e-commerce data with weekly demand cycles.
    """
    folds          = build_folds(len(series))
    fold_metrics   = []
    fold_residuals = []

    for fold in folds:
        tr_s, tr_e = fold["train"]
        te_s, te_e = fold["test"]

        train     = series.iloc[tr_s:tr_e].values
        test      = series.iloc[te_s:te_e].values
        predicted = np.array([
            train[-(7 - (j % 7))] if (7 - (j % 7)) <= len(train) else train[-1]
            for j in range(len(test))
        ])
        predicted = np.clip(predicted, 0, None)

        m = all_metrics(test, predicted, train)
        fold_metrics.append(m)

        for j in range(len(test)):
            fold_residuals.append({
                "date":      dates.iloc[te_s + j],
                "actual":    test[j],
                "predicted": predicted[j],
                "residual":  test[j] - predicted[j],
                "fold":      fold["fold"],
            })

        log.info(f"    Fold {fold['fold']} "
                 f"(train={fold['train_days']}d): "
                 f"MAPE={m['mape']:.1f}%  MASE={m['mase']:.2f}  Bias={m['bias']:+.2f}")

    if not fold_metrics:
        return {"model": "Seasonal_Naive", "mape": None, "n_folds": 0}, []

    avg = {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    avg["model"]   = "Seasonal_Naive"
    avg["n_folds"] = len(fold_metrics)
    return avg, fold_residuals


# ── Baseline 4: Prophet ───────────────────────────────────────────────────────

def run_prophet(series: pd.Series, dates: pd.Series) -> tuple[dict, list[dict]]:
    """
    Prophet with additive seasonality + 90% confidence intervals.
    Additive chosen because 14.4% zero-demand days exist (confirmed in
    category_stats.json) — multiplicative mode produces 0 × factor = 0,
    making seasonal adjustment useless on zero-demand days.

    Coverage metric evaluates interval quality:
      ~90% = well-calibrated
      <80% = overconfident intervals
      >95% = intervals too wide (conservative)
    """
    try:
        from prophet import Prophet
    except ImportError:
        log.error("prophet not installed — pip install prophet")
        return {"model": "Prophet", "mape": None, "n_folds": 0}, []

    folds          = build_folds(len(series))
    fold_metrics   = []
    fold_residuals = []

    for fold in folds:
        tr_s, tr_e = fold["train"]
        te_s, te_e = fold["test"]

        train_df = pd.DataFrame({
            "ds": dates.iloc[tr_s:tr_e].values,
            "y":  series.iloc[tr_s:tr_e].values,
        })
        train_vals  = series.iloc[tr_s:tr_e].values
        test_actual = series.iloc[te_s:te_e].values
        test_dates  = dates.iloc[te_s:te_e].values

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.90,
        )
        model.fit(train_df)

        future   = model.make_future_dataframe(periods=len(test_actual), freq="D")
        forecast = model.predict(future)
        tail     = forecast.tail(len(test_actual))

        predicted = np.clip(tail["yhat"].values,       0, None)
        lower     = np.clip(tail["yhat_lower"].values,  0, None)
        upper     = np.clip(tail["yhat_upper"].values,  0, None)

        m = all_metrics(test_actual, predicted, train_vals, lower, upper)
        fold_metrics.append(m)

        for j in range(len(test_actual)):
            fold_residuals.append({
                "date":      test_dates[j],
                "actual":    test_actual[j],
                "predicted": predicted[j],
                "lower_90":  lower[j],
                "upper_90":  upper[j],
                "residual":  test_actual[j] - predicted[j],
                "fold":      fold["fold"],
            })

        cov = m.get("coverage_90", float("nan"))
        log.info(f"    Fold {fold['fold']} "
                 f"(train={fold['train_days']}d): "
                 f"MAPE={m['mape']:.1f}%  MASE={m['mase']:.2f}  "
                 f"Coverage={cov:.1f}%  Bias={m['bias']:+.2f}")

    if not fold_metrics:
        return {"model": "Prophet", "mape": None, "n_folds": 0}, []

    avg = {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    avg["model"]   = "Prophet"
    avg["n_folds"] = len(fold_metrics)
    return avg, fold_residuals


# ── Residual Analysis ─────────────────────────────────────────────────────────

def analyze_residuals(residuals: list[dict], category: str, model: str) -> dict:
    """
    Computes residual diagnostics and saves a residual plot.

    Plot output structure (pathlib, cross-platform):
        data/plots/{model}/{category}/residuals.png
    """
    if not residuals:
        return {}

    df  = pd.DataFrame(residuals)
    res = df["residual"].values
    act = df["actual"].values

    df["weekday"]  = pd.to_datetime(df["date"]).dt.dayofweek
    weekend_res    = df[df["weekday"] >= 5]["residual"].mean()
    weekday_res    = df[df["weekday"] <  5]["residual"].mean()
    weekend_bias   = float(weekend_res - weekday_res)

    analysis = {
        "mean_residual":     float(np.mean(res)),
        "std_residual":      float(np.std(res)),
        "max_overforecast":  float(np.max(res)),
        "max_underforecast": float(np.min(res)),
        "zero_day_count":    int((act == 0).sum()),
        "pct_within_20pct":  float(np.mean(np.abs(res / (act + 1e-6)) < 0.20) * 100),
        "weekend_bias":      weekend_bias,
        "has_weekend_bias":  abs(weekend_bias) > 1.0,
    }

    if analysis["has_weekend_bias"]:
        direction = "under" if weekend_bias > 0 else "over"
        log.info(f"    Weekend bias: {model} {direction}-forecasts weekends "
                 f"by {abs(weekend_bias):.2f} units avg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(pd.to_datetime(df["date"]), res,
             alpha=0.7, color="steelblue", linewidth=0.8)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1)
    ax1.set_title(f"{model} Residuals — {category}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Actual − Predicted")

    ax2.hist(res, bins=25, color="steelblue", alpha=0.8, edgecolor="white")
    ax2.axvline(0,            color="red",    linestyle="--", linewidth=1)
    ax2.axvline(np.mean(res), color="orange", linestyle="--", linewidth=1,
                label=f"Mean={np.mean(res):.2f}")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual")
    ax2.legend(fontsize=8)

    plt.tight_layout()

    # ── Structured, per-model/per-category plot directory ──
    # data/plots/{model}/{category}/residuals.png
    plot_dir = PLOTS_DIR / model / category
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "residuals.png"

    plt.savefig(plot_path, dpi=100)
    plt.close()

    log.info(f"    Saved residual plot → {plot_path.as_posix()}")

    return analysis


# ── Model Comparison Plot ─────────────────────────────────────────────────────

def create_comparison_plots(results_df: pd.DataFrame) -> None:
    """
    Cross-model summary charts built from results_df (grouped by model):
      - avg_mape_by_model.png
      - avg_mase_by_model.png

    Saved to: data/plots/model_comparison/
    """
    comparison_dir = PLOTS_DIR / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    model_order = ["Last_Value", "Average_Demand", "Seasonal_Naive", "Prophet"]
    grouped = results_df.groupby("model")[["mape", "mase"]].mean()
    grouped = grouped.reindex([m for m in model_order if m in grouped.index])

    # ── Avg MAPE by model ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped.index, grouped["mape"], color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Average MAPE by Model (across all categories)")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAPE (%)")
    if grouped["mape"].max() > 0:
        ax.set_ylim(0, grouped["mape"].max() * 1.2)
    plt.xticks(rotation=15)
    plt.tight_layout()

    mape_path = comparison_dir / "avg_mape_by_model.png"
    plt.savefig(mape_path, dpi=100)
    plt.close()
    log.info(f"Saved comparison plot → {mape_path.as_posix()}")

    # ── Avg MASE by model ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped.index, grouped["mase"], color="darkorange", edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1,
               label="Naive threshold (MASE = 1.0)")
    ax.set_title("Average MASE by Model (across all categories)")
    ax.set_xlabel("Model")
    ax.set_ylabel("MASE")
    ax.legend(fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()

    mase_path = comparison_dir / "avg_mase_by_model.png"
    plt.savefig(mase_path, dpi=100)
    plt.close()
    log.info(f"Saved comparison plot → {mase_path.as_posix()}")


# ── Diebold-Mariano Test ──────────────────────────────────────────────────────

def diebold_mariano(errors_a: list[float], errors_b: list[float],
                    model_a: str, model_b: str) -> dict:
    """
    Tests whether the difference in forecast accuracy between two models
    is statistically significant.
    H0: no difference in forecast accuracy.
    p < 0.05 → reject H0 → one model is statistically better.

    Applied only when both models have actual fold-level errors to compare.
    Only meaningful when model MAPEs are close (within 5%).
    """
    if len(errors_a) < 3 or len(errors_b) < 3:
        return {"dm_pvalue": None, "dm_significant": None}

    diff   = np.array(errors_a) - np.array(errors_b)
    t_stat, p_val = scipy_stats.ttest_1samp(diff, 0)

    result = {
        "dm_model_a":      model_a,
        "dm_model_b":      model_b,
        "dm_tstat":        float(t_stat),
        "dm_pvalue":       float(p_val),
        "dm_significant":  bool(p_val < 0.05),
    }

    if p_val < 0.05:
        better = model_a if t_stat < 0 else model_b
        log.info(f"    DM test: {better} is statistically better "
                 f"(p={p_val:.3f}, t={t_stat:.2f})")
    else:
        log.info(f"    DM test: no significant difference between "
                 f"{model_a} and {model_b} (p={p_val:.3f})")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("BASELINE MODELS")
    log.info("Last-Value | Avg Demand | Seasonal Naive | Prophet")
    log.info(f"Folds: {N_FOLDS} × {TEST_DAYS}-day windows | Min train: {MIN_TRAIN_DAYS}d")
    log.info("=" * 55)

    df, cat_stats = load_data()

    if cat_stats:
        log.info("\nCategory quality summary:")
        for cat, s in cat_stats.items():
            flag = " ⚠ high zeros" if s["zero_pct"] > 40 else ""
            log.info(f"  {cat:<35} zero={s['zero_pct']:.1f}%  "
                     f"avg_daily={s['avg_daily_demand']:.1f}{flag}")

    all_results      = []
    all_prophet_fc   = []
    baseline_summary = {}
    weekend_bias_cats = []

    for cat in df["category"].unique():
        cat_df = (df[df["category"] == cat]
                  .sort_values("sale_date")
                  .reset_index(drop=True))

        n = len(cat_df)
        if n < MIN_TRAIN_DAYS + TEST_DAYS:
            log.warning(f"\n{cat}: only {n} rows — skipping")
            continue

        series = cat_df["units_sold_raw"]
        dates  = cat_df["sale_date"]

        log.info(f"\n── {cat.upper()}  ({n} days) ──")

        # Run all 4 baselines
        log.info("  [1/4] Last-Value:")
        lv_avg, lv_res = run_last_value(series, dates)
        lv_avg["category"] = cat
        all_results.append(lv_avg)
        lv_analysis = analyze_residuals(lv_res, cat, "Last_Value")

        log.info("  [2/4] Average Demand (30d window):")
        ad_avg, ad_res = run_average_demand(series, dates, window=30)
        ad_avg["category"] = cat
        all_results.append(ad_avg)
        ad_analysis = analyze_residuals(ad_res, cat, "Average_Demand")

        log.info("  [3/4] Seasonal Naive:")
        sn_avg, sn_res = run_seasonal_naive(series, dates)
        sn_avg["category"] = cat
        all_results.append(sn_avg)
        sn_analysis = analyze_residuals(sn_res, cat, "Seasonal_Naive")

        log.info("  [4/4] Prophet:")
        pr_avg, pr_res = run_prophet(series, dates)
        pr_avg["category"] = cat
        all_results.append(pr_avg)
        pr_analysis = analyze_residuals(pr_res, cat, "Prophet")

        # Save Prophet confidence intervals for Streamlit forecast band
        if pr_res:
            fc_df = pd.DataFrame(pr_res)
            fc_df["category"] = cat
            all_prophet_fc.append(fc_df)

        # Diebold-Mariano: only run when Seasonal Naive vs Prophet are close
        sn_mape = sn_avg.get("mape") or 999
        pr_mape = pr_avg.get("mape") or 999
        if abs(sn_mape - pr_mape) < 5 and sn_res and pr_res:
            log.info("  DM Test (Seasonal Naive vs Prophet — close MAPEs):")
            sn_fold_maes = [abs(r["residual"]) for r in sn_res]
            pr_fold_maes = [abs(r["residual"]) for r in pr_res]
            dm_result    = diebold_mariano(
                sn_fold_maes, pr_fold_maes, "Seasonal_Naive", "Prophet"
            )
        else:
            dm_result = {}

        # Weekend bias tracking
        if (lv_analysis.get("has_weekend_bias") or
                ad_analysis.get("has_weekend_bias") or
                sn_analysis.get("has_weekend_bias") or
                pr_analysis.get("has_weekend_bias")):
            weekend_bias_cats.append(cat)

        # LSTM target: beat best baseline by 15%
        best_mape   = min(lv_avg.get("mape") or 999,
                          ad_avg.get("mape") or 999,
                          sn_mape,
                          pr_mape)
        lstm_target = best_mape * 0.85

        # Prophet coverage quality flag
        pr_coverage = pr_avg.get("coverage_90")
        if pr_coverage is not None:
            if pr_coverage < 80:
                cov_flag = "OVERCONFIDENT — intervals too narrow"
            elif pr_coverage > 95:
                cov_flag = "CONSERVATIVE — intervals too wide"
            else:
                cov_flag = "WELL-CALIBRATED"
        else:
            cov_flag = "N/A"

        baseline_summary[cat] = {
            "last_value_mape":   lv_avg.get("mape"),
            "avg_demand_mape":   ad_avg.get("mape"),
            "seasonal_naive_mape": sn_mape if sn_mape < 999 else None,
            "prophet_mape":      pr_mape  if pr_mape  < 999 else None,
            "best_baseline":     best_mape,
            "lstm_target_mape":  lstm_target,
            "prophet_coverage":  pr_coverage,
            "prophet_coverage_quality": cov_flag,
            "weekend_bias":      sn_analysis.get("has_weekend_bias", False),
            "zero_pct":          cat_stats.get(cat, {}).get("zero_pct"),
            **dm_result,
        }

        log.info(f"\n  Summary:")
        log.info(f"    Last-Value    : {lv_avg.get('mape', 'N/A'):.1f}%  MASE={lv_avg.get('mase', 0):.2f}")
        log.info(f"    Avg Demand    : {ad_avg.get('mape', 'N/A'):.1f}%  MASE={ad_avg.get('mase', 0):.2f}")
        log.info(f"    Seasonal Naive: {sn_mape:.1f}%  MASE={sn_avg.get('mase', 0):.2f}")
        log.info(f"    Prophet       : {pr_mape:.1f}%  Coverage={pr_coverage or 0:.1f}%  [{cov_flag}]")
        log.info(f"    LSTM target   : <{lstm_target:.1f}% MAPE")

    # ── Save outputs ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(PROCESSED_DIR / "baseline_results.csv", index=False)

    with open(PROCESSED_DIR / "baseline_summary.json", "w") as f:
        json.dump(baseline_summary, f, indent=2, default=str)

    if all_prophet_fc:
        pd.concat(all_prophet_fc, ignore_index=True).to_csv(
            PROCESSED_DIR / "prophet_forecasts.csv", index=False
        )

    # ── Cross-model comparison plots ─────────────────────────────────────────
    create_comparison_plots(results_df)

    # ── Final report ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("FINAL RESULTS SUMMARY")
    log.info("=" * 55)

    pivot = (results_df.groupby("model")[["mape", "mase"]]
             .mean()
             .round(2)
             .rename(columns={"mape": "Avg MAPE%", "mase": "Avg MASE"}))
    log.info(f"\n{pivot.sort_values('Avg MAPE%').to_string()}")

    best_model = results_df.groupby("model")["mape"].mean().idxmin()
    best_mape  = results_df.groupby("model")["mape"].mean().min()

    log.info(f"\nBest baseline : {best_model}  @  {best_mape:.1f}% avg MAPE")
    log.info(f"LSTM must hit : <{best_mape * 0.85:.1f}% to justify complexity")

    # MASE interpretation
    log.info("\nMASE interpretation: <1.0 = beats naive, >1.0 = worse than naive")
    mase_summary = results_df.groupby("model")["mase"].mean().round(2)
    for model, val in mase_summary.items():
        flag = "✓ beats naive" if val < 1.0 else "✗ worse than naive"
        log.info(f"  {model:<20} MASE={val:.2f}  {flag}")

    if weekend_bias_cats:
        log.info(f"\nWeekend bias detected: {weekend_bias_cats}")
        log.info("  is_weekend + dow_sin/cos in LSTM features — should handle this ✓")

    log.info(f"\nOutputs → {PROCESSED_DIR}")
    log.info(f"Plots   → {PLOTS_DIR}")
    log.info("Next    → python forecasting/lstm_model.py")
    log.info("         (or: python forecasting/lstm_model.py --skip-cv for faster run)")


if __name__ == "__main__":
    main()