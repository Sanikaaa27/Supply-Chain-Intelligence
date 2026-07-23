"""
FastAPI Forecast Server.

Endpoints
---------
GET  /health
GET  /categories
POST /forecast   {"category": ..., "model": "auto"|"gbm"|"lstm"}
POST /explain    {"category": ..., "model": "auto", "horizon_day": 1-7}

Change from original:
- /explain now accepts optional `horizon_day` (1–7, default 1).
  All 7 GBM explainers were already built at startup (build_gbm_explainers
  loops over all horizon days) — this just exposes them.
  LSTM still returns 501 for any horizon.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import joblib
import lightgbm as lgb
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting.model_selector import ModelSelector, ModelSelectionUnavailable
from forecasting.inference_features import (
    build_inference_row,
    build_inference_sequence,
    load_scalers,
    InferenceFeatureError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("forecast_api")

MODELS_DIR    = Path(__file__).parent.parent / "data" / "models"
FORECAST_DAYS = 7
PI_LEVELS     = (80, 95)
QUANTILE_ALPHAS = {80: (0.10, 0.90), 95: (0.025, 0.975)}

STATE: dict = {}


def to_raw_scale(values: np.ndarray, target_scaler, clip_negative: bool = False) -> np.ndarray:
    flat = np.asarray(values).reshape(-1, 1)
    if clip_negative:
        flat = np.clip(flat, 0, None)
    inv = target_scaler.inverse_transform(flat).flatten()
    if clip_negative:
        inv = np.clip(inv, 0, None)
    return np.expm1(inv)


def load_gbm_models() -> dict:
    point_models    = {}
    quantile_models = {}
    for day in range(1, FORECAST_DAYS + 1):
        path = MODELS_DIR / f"gbm_smoothed_day{day}.txt"
        if not path.exists():
            raise RuntimeError(f"Missing GBM model: {path}")
        point_models[day] = lgb.Booster(model_file=str(path))

        quantile_models[day] = {}
        for level, (lo_a, hi_a) in QUANTILE_ALPHAS.items():
            for alpha in (lo_a, hi_a):
                qpath = MODELS_DIR / f"gbm_smoothed_day{day}_q{alpha}.txt"
                if qpath.exists():
                    quantile_models[day][alpha] = lgb.Booster(model_file=str(qpath))
                else:
                    log.warning(f"Missing quantile model {qpath.name} — "
                                f"{level}% interval for day {day} unavailable")
    return {"point": point_models, "quantile": quantile_models}


def build_gbm_explainers(gbm_models: dict) -> dict:
    """SHAP TreeExplainer per horizon day (1–7), built once at startup."""
    explainers = {}
    for day, booster in gbm_models["point"].items():
        explainers[day] = shap.TreeExplainer(booster)
    return explainers


def load_lstm_model():
    from tensorflow.keras.models import load_model
    path = MODELS_DIR / "lstm_global_smoothed.h5"
    if not path.exists():
        raise RuntimeError(f"Missing LSTM model: {path}")
    return load_model(str(path), compile=False)


def load_lstm_intervals() -> dict | None:
    path = MODELS_DIR.parent / "processed" / "lstm_global_smoothed_intervals.json"
    if not path.exists():
        log.warning(f"{path} not found — LSTM forecasts will have no prediction intervals.")
        return None
    import json
    with open(path) as f:
        return json.load(f)


async def lifespan(app: FastAPI):
    log.info("Loading models, scalers, and selector at startup...")
    STATE["selector"]       = ModelSelector()
    STATE["target_scaler"], _ = load_scalers()
    STATE["gbm"]            = load_gbm_models()
    STATE["gbm_explainers"] = build_gbm_explainers(STATE["gbm"])
    STATE["lstm"]           = None  # TensorFlow disabled for deployment
    STATE["lstm_intervals"] = None
    log.info("Startup complete — ready to serve forecasts.")
    yield
    STATE.clear()


app = FastAPI(title="Supply Chain Demand Forecast API", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────

class ForecastRequest(BaseModel):
    category: str
    model: str = Field(default="auto", description="'auto', 'gbm', or 'lstm'")


class ExplainRequest(BaseModel):
    category: str
    model: str = Field(default="auto")
    horizon_day: int = Field(
        default=1,
        ge=1,
        le=7,
        description="Which forecast horizon to explain (1 = tomorrow, 7 = 7 days out). "
                    "Only GBM-routed categories support SHAP. Default: 1.",
    )


class IntervalBlock(BaseModel):
    lower: list[float]
    upper: list[float]


class ForecastResponse(BaseModel):
    category:         str
    model_used:       str
    confidence:       str
    selection_reason: str
    forecast:         list[float]
    interval_80:      IntervalBlock | None = None
    interval_95:      IntervalBlock | None = None


class FeatureImpact(BaseModel):
    feature:    str
    shap_value: float
    direction:  str   # "increases" | "decreases"


class ExplainResponse(BaseModel):
    category:     str
    model_used:   str
    horizon_day:  int
    base_value:   float
    top_features: list[FeatureImpact]
    note:         str | None = None


# ── Inference helpers ────────────────────────────────────────────────────

def predict_gbm(category: str) -> dict:
    row          = build_inference_row(category)
    target_scaler = STATE["target_scaler"]
    gbm          = STATE["gbm"]
    feature_cols = list(row.columns)

    point_forecast_scaled = np.array([
        gbm["point"][day].predict(row[feature_cols])[0]
        for day in range(1, FORECAST_DAYS + 1)
    ])
    forecast_raw = to_raw_scale(point_forecast_scaled, target_scaler, clip_negative=True)

    intervals = {}
    for level, (lo_a, hi_a) in QUANTILE_ALPHAS.items():
        if lo_a not in gbm["quantile"][1] or hi_a not in gbm["quantile"][1]:
            continue
        lo_scaled = np.array([gbm["quantile"][day][lo_a].predict(row[feature_cols])[0]
                               for day in range(1, FORECAST_DAYS + 1)])
        hi_scaled = np.array([gbm["quantile"][day][hi_a].predict(row[feature_cols])[0]
                               for day in range(1, FORECAST_DAYS + 1)])
        lo_scaled, hi_scaled = np.minimum(lo_scaled, hi_scaled), np.maximum(lo_scaled, hi_scaled)
        lo_raw = to_raw_scale(lo_scaled, target_scaler, clip_negative=True)
        hi_raw = to_raw_scale(hi_scaled, target_scaler, clip_negative=True)
        lo_raw, hi_raw = np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw)
        intervals[level] = {"lower": lo_raw.tolist(), "upper": hi_raw.tolist()}

    return {"forecast": forecast_raw.tolist(), "intervals": intervals}


def predict_lstm(category: str) -> dict:
    seq           = build_inference_sequence(category)
    target_scaler = STATE["target_scaler"]
    model         = STATE["lstm"]

    pred_delta     = model.predict([seq["X"], seq["category_id"]], verbose=0)[0]
    forecast_scaled = pred_delta + seq["anchor"][0]
    forecast_raw   = to_raw_scale(forecast_scaled, target_scaler, clip_negative=True)

    intervals     = {}
    interval_data = STATE.get("lstm_intervals")
    if interval_data:
        for level in PI_LEVELS:
            lvl_str = str(level)
            try:
                lower = np.array([
                    forecast_raw[d] + interval_data[str(d + 1)][lvl_str]["lower_offset"]
                    for d in range(FORECAST_DAYS)
                ])
                upper = np.array([
                    forecast_raw[d] + interval_data[str(d + 1)][lvl_str]["upper_offset"]
                    for d in range(FORECAST_DAYS)
                ])
                lower = np.clip(lower, 0, None)
                upper = np.clip(upper, 0, None)
                intervals[level] = {"lower": lower.tolist(), "upper": upper.tolist()}
            except KeyError:
                log.warning(f"LSTM interval data missing for level={level} — skipping")

    return {"forecast": forecast_raw.tolist(), "intervals": intervals}


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(STATE.keys())}


@app.get("/categories")
def categories():
    return {
        "known_categories": STATE["selector"].known_categories(),
        "global_default":   STATE["selector"].global_default(),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    selector: ModelSelector = STATE["selector"]

    try:
        resolved = selector.resolve(req.category, req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelSelectionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        if resolved.model == "gbm":
            result = predict_gbm(req.category)
        elif resolved.model == "lstm":
            raise HTTPException(
                status_code=501,
                detail="LSTM unavailable in hosted deployment. Use model='auto' or 'gbm'.",
            )
        else:
            raise HTTPException(
                status_code=501,
                detail="'baseline' model serving not implemented — use 'auto', 'gbm', or 'lstm'.",
            )
    except InferenceFeatureError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ForecastResponse(
        category=req.category,
        model_used=resolved.model,
        confidence=resolved.confidence,
        selection_reason=resolved.reason,
        forecast=result["forecast"],
        interval_80=result["intervals"].get(80),
        interval_95=result["intervals"].get(95),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """SHAP explanation for any GBM forecast horizon (Day 1–7).

    horizon_day controls which of the 7 per-horizon GBM boosters is
    explained. All 7 SHAP explainers are pre-built at startup so there is
    no extra latency cost for later horizons vs Day 1.

    LSTM returns 501 — SHAP is not available for the LSTM architecture used
    here (anchor-delta head with a Keras model). Returning a fake number
    would be worse than returning nothing.

    SHAP values are in SCALED model-output space (pre-inverse-transform,
    pre-expm1). Direction is reliable; exact magnitude in raw units is not.
    """
    selector: ModelSelector = STATE["selector"]

    try:
        resolved = selector.resolve(req.category, req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelSelectionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    if resolved.model != "gbm":
        raise HTTPException(
            status_code=501,
            detail=(
                f"SHAP is only available for GBM-routed categories. "
                f"'{req.category}' resolves to '{resolved.model}'. "
                f"LSTM explainability is not yet supported."
            ),
        )

    try:
        row = build_inference_row(req.category)
    except InferenceFeatureError as e:
        raise HTTPException(status_code=422, detail=str(e))

    feature_cols = list(row.columns)
    explainer    = STATE["gbm_explainers"][req.horizon_day]
    shap_values  = explainer.shap_values(row[feature_cols])[0]
    base_value   = float(explainer.expected_value)

    pairs = sorted(zip(feature_cols, shap_values), key=lambda x: abs(x[1]), reverse=True)[:8]
    top_features = [
        FeatureImpact(
            feature=f,
            shap_value=round(float(v), 5),
            direction="increases" if v > 0 else "decreases",
        )
        for f, v in pairs
    ]

    return ExplainResponse(
        category=req.category,
        model_used="gbm",
        horizon_day=req.horizon_day,
        base_value=base_value,
        top_features=top_features,
        note=(
            f"SHAP values for Day-{req.horizon_day} GBM booster, in scaled model-output space "
            f"(pre-inverse-transform). Direction is reliable; raw magnitude is not in original units."
        ),
    )