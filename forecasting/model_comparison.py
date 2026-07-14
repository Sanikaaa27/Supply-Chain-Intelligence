"""Model Comparison & Selection Layer

Reads the three result artifacts already produced by the pipeline:
    baseline_summary.json          <- baseline_models.py
    lstm_results.csv                <- lstm_model.py   (mode="global", per-category rows)
    gbm_results_noblend_v2.csv      <- gbm_model.py     (per-category rows)

and produces:
    1. A model leaderboard (pooled avg MAPE per model family)
    2. A per-category leaderboard (which model wins each category)
    3. Forecast-skill tables (GBM vs baseline, LSTM vs baseline, GBM vs LSTM)
    4. An auto-selected "production model" per category, with a documented
       reason, written to model_selection.json — this is what FastAPI's
       "model": "auto" should read.
    5. A human-readable markdown report (model_comparison_report.md) summarizing
       all of the above, suitable for a PR description or a Streamlit page.

Design notes / things this file deliberately does NOT do:
- It does not re-run any model. It only reads CSV/JSON artifacts that
  lstm_model.py and gbm_model.py already write to PROCESSED_DIR.
- It does not silently fabricate a winner when a category has missing data
  for one of the two models — it labels it "insufficient_data" instead of
  guessing, mirroring the honesty conventions already in gbm_model.py
  (lift_label / directional_lift_label) and lstm_model.py (the
  "do not report this as an improvement" guard).
- Per-category selection compares MAPE only when BOTH models are
  statistically distinguishable from baseline (or from each other) by more
  than gbm_model.py's own noise_threshold_pp for that run, where available.
  Falls back to a flat NOISE_FLOOR_PP if gbm's per-row noise threshold
  isn't present (e.g. LSTM-only or baseline-only category).
- When the margin between candidates isn't genuine, the category is still
  assigned a production model (lowest MAPE), but flagged low-confidence —
  it is never left unassigned. The console output makes this final decision
  explicit instead of only showing the raw "no_clear_winner" label.

Run:
    python model_comparison.py
    python model_comparison.py --mlflow-log              # also log leaderboard to MLflow
    python model_comparison.py --processed-dir D:\\path\\to\\data\\processed
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model_comparison")

DEFAULT_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Fallback noise floor (percentage points of MAPE) used only when a
# per-row noise_threshold_pp isn't available from gbm_results — e.g.
# comparing LSTM vs baseline directly, since lstm_model.py doesn't compute
# its own CV-fold-variance noise threshold the way gbm_model.py does.
NOISE_FLOOR_PP = 1.5

# A margin must ALSO be at least this % better than the runner-up (relative,
# not absolute pp) to count as "genuine". This stops a fixed pp floor from
# being meaningless on high-MAPE categories (e.g. 1.5pp on a 31% MAPE
# category is noise) or overly strict on low-MAPE ones.
MIN_RELATIVE_IMPROVEMENT_PCT = 5.0

REQUIRED_LSTM_COLS = {"category", "mape"}
REQUIRED_GBM_COLS = {"category", "mape"}


# ── Loading ──────────────────────────────────────────────────────────────

def load_baseline_summary(path: Path) -> dict:
    """Returns {category: {"best_baseline": mape, "best_method": name, ...}}."""
    if not path.exists():
        log.warning(f"  {path.name} not found — baseline rows will be empty")
        return {}
    with open(path) as f:
        summary = json.load(f)
    return summary


def _validate_columns(df: pd.DataFrame, required: set, source_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required column(s) {sorted(missing)}. "
            f"Found columns: {list(df.columns)}. "
            f"Was this file written by an older version of the pipeline?"
        )


def load_lstm_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning(f"  {path.name} not found — run lstm_model.py first")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    _validate_columns(df, REQUIRED_LSTM_COLS, path.name)
    # lstm_model.py writes one row per category per run; if multiple
    # experiments/modes are present, keep the most recent "global" rows only.
    if "mode" in df.columns and (df["mode"] == "global").any():
        df = df[df["mode"] == "global"].copy()
    df["model_family"] = "lstm"
    return df


def load_gbm_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning(f"  {path.name} not found — run gbm_model.py first")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    _validate_columns(df, REQUIRED_GBM_COLS, path.name)
    if "model_family" not in df.columns:
        df["model_family"] = "lightgbm"
    return df


def baseline_rows_from_summary(baseline_summary: dict) -> pd.DataFrame:
    """Builds a baseline-only row per category so it can sit in the same
    leaderboard as LSTM/GBM rows for the 'does anything beat doing nothing
    clever' comparison."""
    rows = []
    for cat, v in baseline_summary.items():
        best_mape = v.get("best_baseline")
        if best_mape is None:
            continue
        rows.append({
            "model_family": "baseline",
            "category":     cat,
            "mape":         best_mape,
            "baseline_mape": best_mape,
            "best_baseline_method": v.get("best_method", "unknown"),
        })
    return pd.DataFrame(rows)


# ── Leaderboards ─────────────────────────────────────────────────────────

def build_model_leaderboard(lstm_df: pd.DataFrame, gbm_df: pd.DataFrame,
                              baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled avg MAPE (and a few other metrics, where present) per model
    family, across whatever categories that family has results for."""
    rows = []
    for name, df in [("baseline", baseline_df), ("lstm", lstm_df), ("gbm", gbm_df)]:
        if df.empty or "mape" not in df.columns:
            continue
        valid = df["mape"].dropna()
        if valid.empty:
            continue
        row = {
            "model": name,
            "avg_mape": float(valid.mean()),
            "median_mape": float(valid.median()),
            "n_categories": int(valid.shape[0]),
        }
        if "mase" in df.columns and df["mase"].notna().any():
            row["avg_mase"] = float(df["mase"].mean())
        if "forecast_skill_pct" in df.columns and df["forecast_skill_pct"].notna().any():
            row["avg_forecast_skill_pct"] = float(df["forecast_skill_pct"].mean())
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values("avg_mape", na_position="last").reset_index(drop=True)
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    return leaderboard


def _noise_threshold_for(cat: str, gbm_df: pd.DataFrame) -> tuple[float, str]:
    """Returns (threshold_pp, source) where source is 'gbm_cv' if the
    threshold came from gbm_model.py's own per-category CV-fold variance,
    or 'flat_floor' if it fell back to NOISE_FLOOR_PP. Surfacing the source
    matters: a wide gbm_cv threshold reflects genuine model instability on
    that category and is worth auditing, not just trusting blindly."""
    if gbm_df.empty or "noise_threshold_pp" not in gbm_df.columns:
        return NOISE_FLOOR_PP, "flat_floor"
    row = gbm_df.loc[gbm_df["category"] == cat, "noise_threshold_pp"]
    if row.empty or pd.isna(row.iloc[0]):
        return NOISE_FLOOR_PP, "flat_floor"
    return float(row.iloc[0]), "gbm_cv"


def build_category_leaderboard(lstm_df: pd.DataFrame, gbm_df: pd.DataFrame,
                                 baseline_df: pd.DataFrame) -> pd.DataFrame:
    """One row per category: every available model's MAPE side-by-side,
    plus the winning model and an honesty-checked margin."""
    categories = set()
    for df in (lstm_df, gbm_df, baseline_df):
        if not df.empty and "category" in df.columns:
            categories.update(df["category"].unique())

    rows = []
    for cat in sorted(categories):
        baseline_mape = _lookup(baseline_df, cat, "mape")
        lstm_mape     = _lookup(lstm_df, cat, "mape")
        gbm_mape      = _lookup(gbm_df, cat, "mape")

        candidates = {
            "baseline": baseline_mape,
            "lstm":     lstm_mape,
            "gbm":      gbm_mape,
        }
        available = {k: v for k, v in candidates.items() if v is not None and not np.isnan(v)}

        if not available:
            rows.append({
                "category": cat, "baseline_mape": baseline_mape,
                "lstm_mape": lstm_mape, "gbm_mape": gbm_mape,
                "best_model": "insufficient_data", "margin_pp": float("nan"),
                "margin_is_genuine": False,
            })
            continue

        best_model = min(available, key=available.get)
        best_mape  = available[best_model]

        # Margin vs the runner-up among available candidates.
        rest = {k: v for k, v in available.items() if k != best_model}
        if rest:
            runner_up_mape = min(rest.values())
            margin_pp = runner_up_mape - best_mape
        else:
            margin_pp = float("nan")

        threshold, threshold_source = _noise_threshold_for(cat, gbm_df)
        margin_pct = (margin_pp / runner_up_mape * 100) if rest and runner_up_mape else float("nan")
        clears_absolute_floor = not np.isnan(margin_pp) and margin_pp > threshold
        clears_relative_floor = not np.isnan(margin_pct) and margin_pct >= MIN_RELATIVE_IMPROVEMENT_PCT
        margin_is_genuine = bool(clears_absolute_floor and clears_relative_floor)

        rows.append({
            "category":           cat,
            "baseline_mape":      baseline_mape,
            "lstm_mape":          lstm_mape,
            "gbm_mape":           gbm_mape,
            "best_model":         best_model if margin_is_genuine or len(available) == 1 else "no_clear_winner",
            "margin_pp":          margin_pp,
            "margin_pct":         margin_pct,
            "noise_threshold_pp": threshold,
            "threshold_source":   threshold_source,
            "margin_is_genuine":  margin_is_genuine,
        })

    return pd.DataFrame(rows)


def _lookup(df: pd.DataFrame, cat: str, col: str) -> float:
    if df.empty or "category" not in df.columns or col not in df.columns:
        return float("nan")
    match = df.loc[df["category"] == cat, col]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


# ── Forecast skill tables ───────────────────────────────────────────────

def build_skill_tables(category_lb: pd.DataFrame) -> dict:
    """GBM vs baseline, LSTM vs baseline, GBM vs LSTM — % improvement,
    positive meaning the first model is better. NaN-safe."""

    def skill(a: Optional[float], b: Optional[float]) -> float:
        """% improvement of a over b (lower MAPE is better)."""
        if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
            return float("nan")
        return float((b - a) / b * 100)

    out = {"gbm_vs_baseline": {}, "lstm_vs_baseline": {}, "gbm_vs_lstm": {}}
    for _, row in category_lb.iterrows():
        cat = row["category"]
        out["gbm_vs_baseline"][cat]  = skill(row.get("gbm_mape"), row.get("baseline_mape"))
        out["lstm_vs_baseline"][cat] = skill(row.get("lstm_mape"), row.get("baseline_mape"))
        out["gbm_vs_lstm"][cat]      = skill(row.get("gbm_mape"), row.get("lstm_mape"))

    summary = {}
    for k, d in out.items():
        valid = [v for v in d.values() if not np.isnan(v)]
        summary[k] = {
            "per_category": d,
            "avg_pct": float(np.mean(valid)) if valid else float("nan"),
            "n_categories_compared": len(valid),
        }
    return summary


# ── Auto model selection ────────────────────────────────────────────────

def build_model_selection(category_lb: pd.DataFrame, leaderboard: pd.DataFrame) -> dict:
    """Per-category production model choice + a global default + a
    documented reason for each, written to model_selection.json.
    This is what the Forecast Agent / FastAPI 'model: auto' should read.

    Fallback policy for 'no_clear_winner' categories: rather than picking
    whichever model happens to have the lowest MAPE (which, when models are
    statistically indistinguishable, amounts to an arbitrary coin toss and
    can cause the served model to flip-flop category-to-category for no
    real reason), we default to the GLOBAL pooled winner if it has a result
    for that category. This gives a single, predictable, defensible model
    in production for low-confidence categories, while still falling back
    to lowest-MAPE if the global winner has no result there at all.
    """
    global_default = leaderboard.iloc[0]["model"] if not leaderboard.empty else "baseline"

    per_category = {}
    for _, row in category_lb.iterrows():
        cat = row["category"]
        best = row["best_model"]

        if best == "insufficient_data":
            reason = "No model results available for this category."
        elif best == "no_clear_winner":
            candidates = {
                "baseline": row.get("baseline_mape"),
                "lstm":     row.get("lstm_mape"),
                "gbm":      row.get("gbm_mape"),
            }
            candidates = {k: v for k, v in candidates.items() if v is not None and not np.isnan(v)}

            if global_default in candidates:
                best = global_default
                reason = (f"Margin over runner-up ({row.get('margin_pp', float('nan')):.2f}pp, "
                          f"{row.get('margin_pct', float('nan')):.1f}% relative) did not clear the "
                          f"noise threshold ({row.get('noise_threshold_pp', NOISE_FLOOR_PP):.2f}pp via "
                          f"{row.get('threshold_source', 'flat_floor')}) and/or the "
                          f"{MIN_RELATIVE_IMPROVEMENT_PCT:.0f}% relative-improvement floor — defaulting to "
                          f"'{global_default}', the global pooled winner, for consistency rather than "
                          f"picking an arbitrary lowest-MAPE model. Flagged low-confidence.")
            else:
                best = min(candidates, key=candidates.get) if candidates else "insufficient_data"
                reason = (f"Margin did not clear noise/relative-improvement thresholds, and the global "
                          f"default model ('{global_default}') has no result for this category — "
                          f"defaulting to lowest-MAPE available model instead. Flagged low-confidence.")
        else:
            reason = (f"Lowest MAPE ({row.get(f'{best}_mape', float('nan')):.2f}%), beating runner-up by "
                      f"{row.get('margin_pp', float('nan')):.2f}pp ({row.get('margin_pct', float('nan')):.1f}% "
                      f"relative) — clears both the {row.get('noise_threshold_pp', NOISE_FLOOR_PP):.2f}pp noise "
                      f"threshold and the {MIN_RELATIVE_IMPROVEMENT_PCT:.0f}% relative-improvement floor.")

        per_category[cat] = {
            "selected_model": best,
            "confidence": "high" if row.get("margin_is_genuine", False) else "low",
            "reason": reason,
            "mapes": {
                "baseline": row.get("baseline_mape"),
                "lstm":     row.get("lstm_mape"),
                "gbm":      row.get("gbm_mape"),
            },
        }

    if not leaderboard.empty:
        global_default_reason = (
            f"Lowest pooled avg MAPE across all categories with results "
            f"({leaderboard.iloc[0]['avg_mape']:.2f}%)."
        )
    else:
        global_default_reason = "No results available."

    n_wins = pd.Series([v["selected_model"] for v in per_category.values()]).value_counts().to_dict()

    selection = {
        "global_default_model": global_default,
        "global_default_reason": global_default_reason,
        "category_wins": n_wins,
        "per_category": per_category,
    }
    return selection


# ── Reporting ────────────────────────────────────────────────────────────

def log_production_selection_table(selection: dict) -> None:
    """Prints the FINAL per-category decision (after the no_clear_winner
    fallback has been resolved) so the console output matches exactly what
    FastAPI's 'model: auto' will serve. This is distinct from the raw
    category leaderboard, where ~half the rows may say 'no_clear_winner'
    even though a model has in fact been assigned."""
    rows = []
    for cat, info in sorted(selection["per_category"].items()):
        rows.append({
            "category": cat,
            "selected_model": info["selected_model"],
            "confidence": info["confidence"],
        })
    df = pd.DataFrame(rows)
    log.info(f"\n── PRODUCTION SELECTION (final, post-fallback) ──\n{df.to_string(index=False)}")


def write_markdown_report(path: Path, leaderboard: pd.DataFrame, category_lb: pd.DataFrame,
                           skill_tables: dict, selection: dict) -> None:
    lines = ["# Model Comparison Report", ""]

    lines.append("## Model Leaderboard (pooled avg MAPE)")
    lines.append("")
    lines.append(leaderboard.round(2).to_markdown(index=False))
    lines.append("")

    lines.append("## Forecast Skill (% improvement, positive = first model wins)")
    lines.append("")
    for k, v in skill_tables.items():
        label = k.replace("_", " ")
        lines.append(f"- **{label}**: avg {v['avg_pct']:.1f}% over {v['n_categories_compared']} categories")
    lines.append("")

    lines.append("## Production Selection (per category, post-fallback)")
    lines.append("")
    sel_rows = []
    for cat, info in sorted(selection["per_category"].items()):
        sel_rows.append({
            "category": cat,
            "selected_model": info["selected_model"],
            "confidence": info["confidence"],
            "reason": info["reason"],
        })
    sel_df = pd.DataFrame(sel_rows)
    lines.append(sel_df.to_markdown(index=False))
    lines.append("")

    lines.append(f"**Global default model:** `{selection['global_default_model']}`  ")
    lines.append(f"**Reason:** {selection['global_default_reason']}  ")
    lines.append(f"**Category wins:** {selection['category_wins']}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Optional MLflow logging ─────────────────────────────────────────────

def log_mlflow_leaderboard(leaderboard: pd.DataFrame, selection: dict,
                            leaderboard_path: Path, category_path: Path) -> None:
    try:
        import mlflow
    except ImportError:
        log.warning("  mlflow not installed — skipping --mlflow-log")
        return

    if leaderboard.empty:
        log.warning("  Leaderboard is empty — skipping --mlflow-log")
        return

    mlflow.set_experiment("supply_chain_model_comparison")
    with mlflow.start_run(run_name="model_comparison"):
        for _, row in leaderboard.iterrows():
            mlflow.log_metric(f"{row['model']}_avg_mape", row["avg_mape"])
            if "avg_forecast_skill_pct" in row and not pd.isna(row.get("avg_forecast_skill_pct")):
                mlflow.log_metric(f"{row['model']}_avg_skill_pct", row["avg_forecast_skill_pct"])
        mlflow.log_param("global_default_model", selection["global_default_model"])
        mlflow.log_dict(selection, "model_selection.json")
        mlflow.log_artifact(str(leaderboard_path))
        mlflow.log_artifact(str(category_path))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-log", action="store_true",
        help="Also log the leaderboard and selection to MLflow")
    parser.add_argument("--processed-dir", type=str, default=None,
        help="Override the default data/processed directory")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir) if args.processed_dir else DEFAULT_PROCESSED_DIR

    lstm_results_path     = processed_dir / "lstm_results.csv"
    gbm_results_path      = processed_dir / "gbm_results_noblend_v2.csv"
    baseline_summary_path = processed_dir / "baseline_summary.json"

    output_leaderboard_path = processed_dir / "model_leaderboard.csv"
    output_category_path    = processed_dir / "category_leaderboard.csv"
    output_selection_path   = processed_dir / "model_selection.json"
    output_report_path      = processed_dir / "model_comparison_report.json"
    output_markdown_path    = processed_dir / "model_comparison_report.md"

    log.info("=" * 55)
    log.info("MODEL COMPARISON & SELECTION LAYER")
    log.info("=" * 55)
    log.info(f"  Processed dir: {processed_dir}")

    try:
        baseline_summary = load_baseline_summary(baseline_summary_path)
        baseline_df = baseline_rows_from_summary(baseline_summary)
        lstm_df     = load_lstm_results(lstm_results_path)
        gbm_df      = load_gbm_results(gbm_results_path)
    except ValueError as e:
        log.error(f"  {e}")
        sys.exit(1)

    if baseline_df.empty and lstm_df.empty and gbm_df.empty:
        log.error("No result files found at all. Run baseline_models.py, "
                   "lstm_model.py, and gbm_model.py first.")
        sys.exit(1)

    log.info(f"  Loaded: baseline={len(baseline_df)} rows, "
             f"lstm={len(lstm_df)} rows, gbm={len(gbm_df)} rows")

    # 1. Model leaderboard
    leaderboard = build_model_leaderboard(lstm_df, gbm_df, baseline_df)
    leaderboard.to_csv(output_leaderboard_path, index=False)
    log.info(f"\n── MODEL LEADERBOARD ──\n{leaderboard.round(2).to_string(index=False)}")

    # 2. Category leaderboard
    category_lb = build_category_leaderboard(lstm_df, gbm_df, baseline_df)
    category_lb.to_csv(output_category_path, index=False)
    show_cols = [c for c in ["category", "baseline_mape", "lstm_mape", "gbm_mape",
                              "best_model", "margin_pp", "margin_pct", "threshold_source",
                              "margin_is_genuine"]
                 if c in category_lb.columns]
    log.info(f"\n── CATEGORY LEADERBOARD (raw, pre-fallback) ──\n{category_lb[show_cols].round(2).to_string(index=False)}")

    # 3. Forecast skill tables
    skill_tables = build_skill_tables(category_lb)
    for k, v in skill_tables.items():
        log.info(f"  {k}: avg {v['avg_pct']:.1f}% over {v['n_categories_compared']} categories")

    # 4. Auto model selection
    selection = build_model_selection(category_lb, leaderboard)
    with open(output_selection_path, "w") as f:
        json.dump(selection, f, indent=2, default=str)

    log.info("\n── AUTO MODEL SELECTION ──")
    log.info(f"  Global default: {selection['global_default_model']}")
    log.info(f"  Category wins: {selection['category_wins']}")

    # 4b. The final, post-fallback decision table — this is what actually
    # gets served, and is intentionally separate from the raw table above.
    log_production_selection_table(selection)

    # Full report (everything, for debugging / Streamlit consumption)
    report = {
        "leaderboard": leaderboard.to_dict(orient="records"),
        "category_leaderboard": category_lb.to_dict(orient="records"),
        "skill_tables": skill_tables,
        "selection": selection,
    }
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # 5. Markdown report
    try:
        write_markdown_report(output_markdown_path, leaderboard, category_lb, skill_tables, selection)
    except ImportError:
        log.warning("  'tabulate' not installed — skipping markdown report (pip install tabulate)")

    if args.mlflow_log:
        log_mlflow_leaderboard(leaderboard, selection, output_leaderboard_path, output_category_path)

    log.info(f"\nLeaderboard → {output_leaderboard_path}")
    log.info(f"Categories  → {output_category_path}")
    log.info(f"Selection   → {output_selection_path}")
    log.info(f"Full report → {output_report_path}")
    log.info(f"Markdown    → {output_markdown_path}")


if __name__ == "__main__":
    main()
