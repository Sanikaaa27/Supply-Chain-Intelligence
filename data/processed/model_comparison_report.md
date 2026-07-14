# Model Comparison Report

## Model Leaderboard (pooled avg MAPE)

|   rank | model    |   avg_mape |   median_mape |   n_categories |   avg_mase |   avg_forecast_skill_pct |
|-------:|:---------|-----------:|--------------:|---------------:|-----------:|-------------------------:|
|      1 | gbm      |      18.56 |         17.52 |             10 |       3.85 |                     6.21 |
|      2 | baseline |      19.17 |         17.61 |             10 |     nan    |                   nan    |
|      3 | lstm     |      20.44 |         20.04 |             10 |       4.04 |                    -3.59 |

## Forecast Skill (% improvement, positive = first model wins)

- **gbm vs baseline**: avg -1.4% over 10 categories
- **lstm vs baseline**: avg -14.1% over 10 categories
- **gbm vs lstm**: avg 9.5% over 10 categories

## Production Selection (per category, post-fallback)

| category              | selected_model   | confidence   | reason                                                                                                                                                                                                                                                                                     |
|:----------------------|:-----------------|:-------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| auto                  | gbm              | low          | Margin over runner-up (1.41pp, 8.2% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence.  |
| bed_bath_table        | gbm              | low          | Margin over runner-up (3.59pp, 20.6% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence. |
| computers_accessories | gbm              | low          | Margin over runner-up (1.65pp, 9.3% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence.  |
| furniture_decor       | gbm              | low          | Margin over runner-up (0.06pp, 0.3% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence.  |
| garden_tools          | gbm              | low          | Margin over runner-up (1.71pp, 6.8% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence.  |
| health_beauty         | gbm              | low          | Margin over runner-up (4.76pp, 27.0% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence. |
| housewares            | gbm              | low          | Margin over runner-up (5.00pp, 26.6% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence. |
| sports_leisure        | gbm              | low          | Margin over runner-up (2.31pp, 13.2% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence. |
| telephony             | lstm             | high         | Lowest MAPE (21.08%), beating runner-up by 5.88pp (21.8% relative) — clears both the 5.15pp noise threshold and the 5% relative-improvement floor.                                                                                                                                         |
| watches_gifts         | gbm              | low          | Margin over runner-up (2.12pp, 11.5% relative) did not clear the noise threshold (5.15pp via gbm_cv) and/or the 5% relative-improvement floor — defaulting to 'gbm', the global pooled winner, for consistency rather than picking an arbitrary lowest-MAPE model. Flagged low-confidence. |

**Global default model:** `gbm`  
**Reason:** Lowest pooled avg MAPE across all categories with results (18.56%).  
**Category wins:** {'gbm': 9, 'lstm': 1}
