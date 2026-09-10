# Model bundle

`nfl_td_model_v2.joblib` is the trained model used by `live/predict_td.py`.

The bundle contains:

- the selected Random Forest pipeline
- the exact 30-feature list
- model type metadata
- validation and holdout metrics for Random Forest, HistGradientBoosting and XGBoost

Random Forest was selected using the 2024 validation season. The 2025 season was kept as the final holdout and was not used to switch models afterward.

You can regenerate the bundle with:

```bash
python training/train_td_model.py
```
