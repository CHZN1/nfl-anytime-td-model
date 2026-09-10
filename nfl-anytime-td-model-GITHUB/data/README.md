# Data pipeline

The repository does not ship the full SQLite database. Rebuild it locally with:

```bash
python data/build_nfl_database.py
python data/validate_database.py
```

`build_nfl_database.py` collects ESPN schedule, box-score and game-context data and builds play-by-play opportunity features with nflverse data accessed through `nfl_data_py`. ESPN tables cover 2020-2025; the published model uses play-by-play features from 2021-2025.

The trainer requires `player_game_logs`, `game_context`, and `pbp_td_features`. Other historical tables may be present but are not required by the current leak-controlled model.

`build_rookie_projections.py` creates optional Week 1 usage priors for drafted RB/WR/TE players using draft capital and depth-chart role. It writes `data/rookie_touch_projections.csv`, which is intentionally ignored by Git because it is generated data.

`nfl_id_crosswalk.csv`, the SQLite database, and parquet caches are also generated locally and ignored by Git.

Historical weather values in the database builder are venue/city proxies rather than observed game-time weather.
