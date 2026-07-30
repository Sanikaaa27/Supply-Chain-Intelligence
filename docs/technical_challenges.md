# 🔥 Technical Challenges Solved

Detailed write-up of real bugs found and fixed during development of the Supply Chain & Demand Intelligence Platform. Moved out of the main README to keep it recruiter-scannable — see the [main README](../README.md) for the project overview.

---

### 1. Log-Transform Mismatch → Inverse-Transform Explosion
**Problem:** `log1p` was applied to the target during feature engineering but not consistently reversed at inference, causing predictions to explode in raw scale.
**Solution:** Single `to_raw_scale()` helper enforced everywhere; direct `inverse_transform()` calls banned outside it.
**Result:** Forecasts back in correct units, verified against known category-level demand ranges.

### 2. Silent MASE Bug (NaN Metrics)
**Problem:** Pooled MASE calculation printed as `nan` because a required `train` argument was missing from one aggregation path — invisible unless you checked the raw output.
**Solution:** `train_raw` is now concatenated and passed through identically to the per-category path.
**Result:** MASE numbers are now real and reported for both GBM (3.85) and LSTM (4.04).

### 3. Quantile Crossing in Prediction Intervals
**Problem:** Independently-trained low/high quantile models could predict `lo > hi`, silently corrupting interval width.
**Solution:** `lo`/`hi` sorted per-row via `np.minimum`/`np.maximum` before any width or coverage calculation.

### 4. Miscalibrated Prediction Intervals
**Problem:** Raw quantile-regression intervals covered only ~64.7% of outcomes at the nominal 80% level, and ~87.2% at the nominal 95% level.
**Solution:** Split-conformal calibration — intervals widened post-hoc using empirical residual quantiles from a held-out validation slice.
**Result:** Both raw and corrected coverage are reported side by side, so the improvement is provable, not asserted.

### 5. Hardcoded "Is This a Real Win?" Threshold
**Problem:** The threshold for calling a model comparison margin "genuine" was a hardcoded 1.5pp guess with no justification.
**Solution:** Threshold is now derived per-run from the actual fold-to-fold MAPE standard deviation across categories (~5.15pp in this run).
**Result:** The answer to "why 5.15pp?" is now a defensible statistical one, not "I picked it."

### 6. Pooled SHAP Hid a Category-Level Failure
**Problem:** SHAP importance computed on a pooled sample across categories masked `telephony`'s very different importance profile — the exact category where the model underperforms.
**Solution:** Per-category top-5 SHAP added for the worst-performing category by lift gap.

### 7. Leakage-Safe Walk-Forward Validation for the LSTM
**Problem:** A global LSTM across 10 categories risks leaking future information across category boundaries if splits aren't handled carefully.
**Solution:** Every split — CV folds, final training, Optuna search — is built **per category, chronologically**, before any pooling; only the pooled *training* set is shuffled (for batch diversity), never the validation/test order.

### 8. Graceful Degradation When the Production DB Is Unreachable
**Problem:** The free-tier Render deployment doesn't keep a live MySQL connection to the analytics layer, so a naive API would simply fail every `/forecast` call in production.
**Solution:** The API detects the unavailable connection and falls back to a precomputed `inference_cache.csv` snapshot instead of erroring out.
**Result:** Verified in production logs — `MySQL unavailable — falling back to inference_cache.csv` followed immediately by a `200 OK` on every `/forecast` call, rather than a `500`.

### 9. Render Free-Tier Cold Starts Breaking the Dashboard's First Request
**Problem:** Render's free tier spins the API down after ~15 minutes of inactivity. The dashboard's initial `/categories` check used a 10-second timeout, which is shorter than a typical cold-start time — causing a hard "API OFFLINE" error on the first load after idle.
**Solution:** A keep-alive cron ping (every 10 minutes) to `/health` prevents the service from ever going idle long enough to spin down; the dashboard's fetch also retries with a "waking up" message as a fallback.
**Result:** No more cold-start failures in normal use.
