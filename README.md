# NFL Anytime Touchdown Model

This project estimates the probability that an NFL player scores at least one rushing or receiving touchdown in a game.

The current model is trained on 2021-2023, uses 2024 for model selection, and keeps 2025 as the final holdout. Predictions from the 2026 season are saved before kickoff and graded afterward so live results stay prospective rather than being used to rewrite the historical test.

## Model selection

Three model families were compared on the 2024 validation season. Random Forest was selected because it produced the best validation ROC-AUC and PR-AUC. The 2025 holdout was scored only after model selection.

| Model | 2024 ROC-AUC | 2024 PR-AUC | 2025 ROC-AUC | 2025 PR-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.6966 | 0.3950 | 0.6876 | 0.3706 |
| HistGradientBoosting | 0.6957 | 0.3934 | 0.6933 | 0.3782 |
| XGBoost | 0.6958 | 0.3916 | 0.6905 | 0.3735 |

HistGradientBoosting happened to score slightly better on the 2025 holdout, but the model was not switched after looking at the holdout.

## What the model uses

The saved bundle contains 30 leak-controlled features built from prior-game information only. The strongest signals are recent player opportunity and touchdown usage, including rolling touches, targets, red-zone opportunity and prior touchdown production. Game context and opponent touchdown allowance are also included.

The training pipeline intentionally avoids full-season aggregates that would leak future information into early-season rows. Per-game play-by-play features are rolled with a one-game shift so the current game's realized usage never enters its own prediction.

## Repository flow

```text
data/build_nfl_database.py
        ↓
data/validate_database.py
        ↓
training/train_td_model.py
        ↓
models/nfl_td_model_v2.joblib
        ↓
data/build_rookie_projections.py   (optional Week 1 rookie priors)
        ↓
live/predict_td.py
        ↓
predictions/nfl_td_predictions_<season>_weekXX.csv
predictions/nfl_td_parlays_<season>_weekXX.csv
        ↓
grading/grade_predictions.py
        ↓
results/
```

## Live prediction behavior

`live/predict_td.py` rebuilds the 30 model features from the local SQLite database, resolves the current slate from ESPN, uses current ESPN rosters for team and position identity, filters confirmed unavailable players with Sleeper status data, fetches anytime-TD prices from The Odds API, and saves the exact pregame decision state for later grading.

Before a season has completed games, veterans fall back to their most recent completed-season form while keeping their current roster identity. True rookie rows can optionally be seeded from `data/rookie_touch_projections.csv`; those rookie priors are shown in the projection table but are excluded from automatic bets and parlays until real NFL history exists.

The live script uses a small disk cache for ESPN slate/rosters, Sleeper player status, and Odds API responses. Odds are cached for a much shorter period than rosters.

### Single-bet thresholds

The model must first have at least a 15% TD probability and at least a 5% relative edge versus the book. The final edge floor then depends on model probability:

| Model P(TD) | Minimum relative edge |
|---|---:|
| 58%+ | 10% |
| 50%-57.9% | 6% |
| 35%-49.9% | 5% |
| 15%-34.9% | 8% |

Qualified bets use 1/8 Kelly with confidence scaling and a 5% bankroll cap.

### Parlays

The current parlay code is **not a correlated Monte Carlo simulator**. It filters qualified, priced legs with `P(TD) >= 25%`, evaluates six-leg combinations from the top candidate pool, and reports the highest joint-probability and highest-EV combinations. Joint probability applies a fixed conservative 0.75 haircut after multiplying leg probabilities. Treat parlay output as experimental until enough prospective results are collected.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and add your Odds API key:

```bash
cp .env.example .env
```

The repository includes the trained model bundle, but not the full SQLite database. Build the database from source data:

```bash
python data/build_nfl_database.py
python data/validate_database.py
```

Optionally build rookie Week 1 priors:

```bash
python data/build_rookie_projections.py --season 2026
```

Run live predictions:

```bash
python live/predict_td.py
```

Retrain from scratch if desired:

```bash
python training/train_td_model.py
```

Grade a completed week after all games are final:

```bash
python grading/grade_predictions.py --season 2026 --week 1
```

## Data sources

The data builder combines ESPN schedule/box-score/context data with nflverse data accessed through `nfl_data_py` for play-by-play, NGS receiving metrics and snap information. Historical weather values in the database builder are venue/city proxies, not observed game-time weather.

Current live team and position identity comes from ESPN rosters. Sleeper is used only as an availability filter. Sportsbook anytime-TD prices come from The Odds API.

## Reproducibility and limitations

- The database itself is excluded from Git because it is generated data.
- The bundled model was selected on 2024 and evaluated once on 2025.
- 2026 results should be treated as prospective monitoring, not a backtest.
- Early-season fallback form is inherently less current than in-season rolling form.
- Rookie priors are conservative usage seeds, not learned TD predictions.
- Injury/status feeds can change close to kickoff; rerun the predictor near the desired betting time.
- Odds move, so saved weekly CSVs are the source of truth for what the script actually saw before games.
- Week-specific prediction and grading CSVs are intended to be committed so prospective performance is auditable over time.

## License

MIT License. See `LICENSE`.
