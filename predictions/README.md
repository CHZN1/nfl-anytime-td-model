# Prediction snapshots

`live/predict_td.py` writes frozen pregame snapshots here.

Typical files:

```text
nfl_td_predictions_2026_week01.csv
nfl_td_parlays_2026_week01.csv
nfl_td_predictions_2026.csv
```

The week-specific files preserve the exact probability, price, edge, bet flag, Kelly stake, confidence, form source and parlay legs seen before games. They are the source of truth for prospective grading.

Generated CSVs are ignored by Git by default.
