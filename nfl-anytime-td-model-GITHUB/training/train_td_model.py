#!/usr/bin/env python3
"""
NFL Rush+Receiving anytime-TD model — leak-free rewrite (v2)
============================================================
Predicts P(total_tds >= 1) per player-game for prop lines. Full rebuild of the
prior trainer, which leaked three ways and joined tables that don't actually
share keys.

WHAT CHANGED vs v1, and WHY
---------------------------
1. pbp_td_features join + as-of-week rolling.
   pbp_td_features holds PER-GAME values (targets, ez_targets, sum_td_prob, NGS,
   offense_pct, ...). v1 never used it. We join it, then for every pbp feature we
   take the player's mean over PRIOR games only (expanding().mean().shift(1)),
   so week W sees weeks 1..W-1. Nothing from the current game enters X.

   JOIN KEY CORRECTION: the two tables do NOT share game_id.
     player_game_logs.game_id = ESPN int   (401547353)
     pbp_td_features.game_id  = nfl_data_py str ('2021_03_GB_SF')  -> 0 overlap.
   But player_id IS shared (both ESPN; 1040/1157 pbp players match) and pbp rows
   carry season + an encodable week. So we bridge on (player_id, season, week).
   Verified unique per role, 73% key coverage (remainder is 2025, absent from PGL).

2. Both fillna(0) calls removed. Missing pbp/rolling values stay NaN so the tree
   models branch on "unknown" (rookie / Week 1) instead of a fake zero that reads
   as "had the ball zero times". HistGBM and XGBoost consume NaN natively; RF gets
   an explicit NaN-safe path (median impute + missing-indicator) so it stays
   comparable without secretly seeing zeros.

3. The three seasonal tables were the biggest leak. receiving_advanced_stats,
   receiving_red_zone_stats, defense_vs_position_stats are keyed (player/team,
   season) with FULL-SEASON aggregates — so a Week 1 row already knew the player's
   whole-season red-zone rate, i.e. the future. v1's season_rz_* block was a
   direct (player_name, season) merge of end-of-year totals. We cannot make those
   pre-agg'd season tables as-of-week (the per-game detail is gone), so the fix is:
   REBUILD the equivalent signal from pbp_td_features per-game rows, rolled
   as-of-week. Red-zone volume/share now comes from ez_targets / inside5_* rolled
   through week W-1. Opponent defense is rebuilt as the opponent's points allowed
   to the position, rolled as-of-week from player_game_logs itself. The static
   season tables are dropped from the feature set entirely.

4. Leak audit (new). Before training we flag any feature that is (a) constant
   within a (player_id, season) — the signature of a season-agg leak — or (b)
   correlated with the same-game target above a threshold that no legitimately
   lagged feature should reach. Audit prints and, for const-within-season columns,
   hard-drops them.

5. Multi-model search: RF / HistGBM / XGBoost, each with an Optuna study, NaN
   preserved, evaluated on a strict SEASON split — train 2021-2023, val 2024,
   test 2025 held out and scored once. (PGL lacks 2025, so pbp-only feature rows
   with no label are dropped; if 2025 labels are unavailable the test slice is
   reported as empty rather than faked.) Permutation importance is printed with an
   explicit watch on sum_td_prob / rush_sum_td_prob: those are nfl_data_py's own
   in-house TD model summed over the game. Rolled as-of-week they're a legitimate
   prior-form signal, but if permutation shows them dominating, we're partly
   predicting nfl_data_py's model rather than TDs — so they're reported separately
   and can be dropped with INCLUDE_NFLDP_TDPROB=False.

Base rate: anytime-TD among active players ~21%, so AUC and PR-AUC (not accuracy)
are the metrics; a naive all-"no" baseline scores 79% accuracy and is useless.
"""

import os
import sqlite3
import warnings
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nfl_td")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.getenv("NFL_TD_DB", str(PROJECT_ROOT / "data" / "nfl_props_COMPLETE.db")))
DEFAULT_MODEL_OUT = Path(os.getenv("NFL_TD_MODEL_OUT", str(PROJECT_ROOT / "models" / "nfl_td_model_v2.joblib")))
TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASON = 2024
TEST_SEASON = 2025
N_TRIALS = 40
INCLUDE_NFLDP_TDPROB = True   # flip to False if permutation shows td_prob dominating
NFLDP_TDPROB_COLS = ["r_sum_td_prob", "r_rush_sum_td_prob"]

# pbp per-game columns to roll as-of-week. Everything here is per-GAME in the
# table and must be lagged before it can be a feature.
PBP_ROLL_COLS = [
    "targets", "ez_targets", "inside10_targets", "inside5_targets",
    "air_yards_game", "adot_game", "sum_td_prob", "ez_target_share",
    "carries", "ez_carries", "inside5_carries", "rush_sum_td_prob",
    "ngs_separation", "ngs_cushion", "ngs_intended_air", "ngs_pct_air_share",
    "offense_pct",
]


# ── data load ──────────────────────────────────────────────────────────────
def load_base(conn):
    """player_game_logs joined to game_context, labelled, one row per player-game."""
    q = """
        SELECT p.player_id, p.player_name, p.game_id, p.season, p.week,
               p.team_id, p.opponent_id, p.game_date,
               COALESCE(p.rushing_tds,0)   AS rushing_tds,
               COALESCE(p.receiving_tds,0) AS receiving_tds,
               COALESCE(p.rushing_attempts,0) AS rushing_attempts,
               COALESCE(p.rushing_yards,0)    AS rushing_yards,
               COALESCE(p.receptions,0)       AS receptions,
               COALESCE(p.receiving_yards,0)  AS receiving_yards,
               COALESCE(p.targets,0)          AS targets_box,
               COALESCE(p.yards_per_carry,0)  AS yards_per_carry,
               COALESCE(p.usage_rate,0)       AS usage_rate,
               COALESCE(p.target_share,0)     AS target_share,
               g.temperature, g.dome_game, g.weather_impact_score,
               g.point_spread, g.over_under
        FROM player_game_logs p
        JOIN game_context g ON p.game_id = g.game_id
        WHERE (p.rushing_attempts >= 1 OR p.targets >= 1)
        ORDER BY p.season, p.week, p.game_id
    """
    df = pd.read_sql_query(q, conn)
    df["total_tds"] = df["rushing_tds"] + df["receiving_tds"]
    df["y"] = (df["total_tds"] >= 1).astype(int)
    # a clean chronological order key for rolling
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce", utc=True)
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return df


def load_pbp(conn):
    """pbp_td_features collapsed to one row per (player_id, season, week).

    A player can appear as both rec and rush in one game (two rows); we sum the
    additive volume columns and take the max/first of the rate columns so the
    week has a single feature vector. Week is decoded from the nfl_data_py
    game_id string ('2021_03_GB_SF' -> 3), which is how we bridge to PGL since
    the two game_id spaces don't match.
    """
    pbp = pd.read_sql_query("SELECT * FROM pbp_td_features", conn)
    pbp["week"] = pbp["game_id"].str.split("_").str[1].astype(int)
    additive = ["targets", "ez_targets", "inside10_targets", "inside5_targets",
                "air_yards_game", "sum_td_prob", "carries", "ez_carries",
                "inside5_carries", "rush_sum_td_prob"]
    rate = ["adot_game", "ez_target_share", "ngs_separation", "ngs_cushion",
            "ngs_intended_air", "ngs_pct_air_share", "offense_pct"]
    agg = {c: "sum" for c in additive if c in pbp.columns}
    agg.update({c: "max" for c in rate if c in pbp.columns})
    g = pbp.groupby(["player_id", "season", "week"], as_index=False).agg(agg)
    return g


# ── as-of-week rolling (the leak fix) ────────────────────────────────────────
def roll_asof(df, cols, group=("player_id",), prefix="r_"):
    """Expanding mean of PRIOR games only. shift(1) drops the current game so
    week W is built from weeks < W. Output columns get the prefix; NaN for a
    player's first game is intentional (no prior history)."""
    out = {}
    grp = df.groupby(list(group))
    for c in cols:
        if c not in df.columns:
            continue
        rolled = grp[c].apply(lambda s: s.expanding().mean().shift(1))
        out[prefix + c] = rolled.reset_index(level=list(range(len(group))), drop=True)
    return pd.DataFrame(out, index=df.index)


def build_opponent_defense_asof(base):
    """Rebuild 'defense vs position' as-of-week from PGL itself, replacing the
    leaky season-agg table. For each (opponent, season, week) we want how many
    rush+rec TDs that defense had allowed PER GAME through the prior week.
    Computed on the defense's game timeline, shifted, then joined back by the
    opponent the offensive player faced."""
    # TDs the team ALLOWED = TDs scored by players whose opponent_id == that team
    allowed = (base.groupby(["opponent_id", "season", "week"], as_index=False)
                    ["total_tds"].sum()
                    .rename(columns={"opponent_id": "def_team", "total_tds": "tds_allowed"}))
    allowed = allowed.sort_values(["def_team", "season", "week"])
    allowed["def_tds_allowed_asof"] = (
        allowed.groupby(["def_team", "season"])["tds_allowed"]
               .apply(lambda s: s.expanding().mean().shift(1))
               .reset_index(level=[0, 1], drop=True))
    j = base.merge(
        allowed[["def_team", "season", "week", "def_tds_allowed_asof"]],
        left_on=["opponent_id", "season", "week"],
        right_on=["def_team", "season", "week"], how="left")
    return j["def_tds_allowed_asof"].values


# ── leak audit ───────────────────────────────────────────────────────────────
def leak_audit(df, feature_cols, target="y", const_frac=0.999, corr_thresh=0.6):
    """Flag season-agg leaks (feature constant within player-season) and any
    feature whose SAME-GAME correlation with the target is implausibly high for
    something that's supposed to be lagged."""
    log.info("── LEAK AUDIT ──")
    const_leak = []
    for c in feature_cols:
        # fraction of (player_id, season) groups where the feature never varies
        g = df.groupby(["player_id", "season"])[c].nunique(dropna=True)
        frac_const = (g <= 1).mean()
        if frac_const >= const_frac and df[c].nunique(dropna=True) > 1:
            const_leak.append((c, frac_const))
    if const_leak:
        log.warning("Constant-within-player-season (season-agg leak signature):")
        for c, f in const_leak:
            log.warning(f"    {c:<28} constant in {f*100:.1f}% of player-seasons -> DROP")
    else:
        log.info("No constant-within-player-season features.")

    corr_flags = []
    y = df[target]
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 100:
            continue
        r = np.corrcoef(s.fillna(s.median()), y)[0, 1]
        if abs(r) >= corr_thresh:
            corr_flags.append((c, r))
    if corr_flags:
        log.warning(f"Same-game corr >= {corr_thresh} (inspect for leakage):")
        for c, r in sorted(corr_flags, key=lambda t: -abs(t[1])):
            log.warning(f"    {c:<28} r={r:+.3f}")
    else:
        log.info(f"No feature exceeds |corr|={corr_thresh} with same-game target.")

    drop = [c for c, _ in const_leak]
    return drop


# ── model search ─────────────────────────────────────────────────────────────
def make_rf(params):
    """RF can't take NaN; give it a NaN-safe pipeline (median impute + missing
    flag) so it's honestly comparable to the native-NaN learners."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestClassifier(random_state=42, n_jobs=-1, **params)),
    ])


def optuna_search(model_type, Xtr, ytr, Xval, yval):
    def objective(trial):
        if model_type == "xgboost":
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 700),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0, 3),
                reg_lambda=trial.suggest_float("reg_lambda", 0, 3),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                random_state=42, eval_metric="logloss", n_jobs=-1, tree_method="hist",
            )
            m = xgb.XGBClassifier(**p)                      # NaN native
            m.fit(Xtr, ytr)
        elif model_type == "histgbm":
            p = dict(
                max_iter=trial.suggest_int("max_iter", 200, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                l2_regularization=trial.suggest_float("l2_regularization", 0, 3),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
                random_state=42,
            )
            m = HistGradientBoostingClassifier(**p)         # NaN native
            m.fit(Xtr, ytr)
        else:  # random_forest
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 600),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            )
            m = make_rf(p)
            m.fit(Xtr, ytr)
        return roc_auc_score(yval, m.predict_proba(Xval)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def fit_final(model_type, params, Xtr, ytr):
    if model_type == "xgboost":
        return xgb.XGBClassifier(random_state=42, eval_metric="logloss",
                                 n_jobs=-1, tree_method="hist", **params).fit(Xtr, ytr)
    if model_type == "histgbm":
        return HistGradientBoostingClassifier(random_state=42, **params).fit(Xtr, ytr)
    return make_rf(params).fit(Xtr, ytr)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    global N_TRIALS
    parser = argparse.ArgumentParser(description="Train the NFL anytime-TD model.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    N_TRIALS = args.trials

    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    base = load_base(conn)
    pbp = load_pbp(conn)
    conn.close()
    log.info(f"base player-games: {len(base):,}   pbp player-weeks: {len(pbp):,}")

    # bridge pbp on (player_id, season, week) — NOT game_id (incompatible spaces)
    merged = base.merge(pbp, on=["player_id", "season", "week"], how="left",
                        suffixes=("", "_pbp"))
    hit = merged["targets"].notna().mean() if "targets" in merged else 0.0
    log.info(f"pbp matched on (player_id,season,week): {hit*100:.0f}% of player-games")

    # as-of-week rolling of pbp per-game features (leak-free)
    merged = merged.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    rolled = roll_asof(merged, PBP_ROLL_COLS, group=("player_id",), prefix="r_")
    merged = pd.concat([merged, rolled], axis=1)

    # as-of-week box-score form from PGL (also lagged; no fillna)
    merged["total_touches"] = merged["rushing_attempts"] + merged["targets_box"]
    for w in (3, 5, 8):
        merged[f"roll{w}_tds"] = (merged.groupby("player_id")["total_tds"]
                                        .apply(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
                                        .reset_index(level=0, drop=True))
        merged[f"roll{w}_touch"] = (merged.groupby("player_id")["total_touches"]
                                          .apply(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
                                          .reset_index(level=0, drop=True))
    merged["td_momentum"] = (merged.groupby("player_id")["total_tds"]
                                   .apply(lambda s: s.rolling(3, min_periods=1).sum().shift(1))
                                   .reset_index(level=0, drop=True))

    # opponent defense rebuilt as-of-week (replaces leaky defense_vs_position_stats)
    merged["def_tds_allowed_asof"] = build_opponent_defense_asof(merged)

    # feature set: rolled pbp + rolled box + as-of-week defense + static game context.
    # NOTE: current-game volume (targets_box, rushing_attempts, ...) is deliberately
    # EXCLUDED — at prediction time on an upcoming game you don't have it, and
    # including the realized snap/target count is itself a soft leak of game script.
    pbp_feats = [f"r_{c}" for c in PBP_ROLL_COLS if f"r_{c}" in merged.columns]
    box_feats = ["roll3_tds", "roll5_tds", "roll8_tds",
                 "roll3_touch", "roll5_touch", "roll8_touch", "td_momentum"]
    ctx_feats = ["temperature", "dome_game", "weather_impact_score",
                 "point_spread", "over_under", "def_tds_allowed_asof"]
    feature_cols = pbp_feats + box_feats + ctx_feats

    if not INCLUDE_NFLDP_TDPROB:
        feature_cols = [c for c in feature_cols if c not in NFLDP_TDPROB_COLS]
        log.info("nfl_data_py td_prob columns EXCLUDED by config.")

    # audit before training; hard-drop constant-within-season leaks
    drop = leak_audit(merged, feature_cols)
    feature_cols = [c for c in feature_cols if c not in drop]
    log.info(f"final feature count: {len(feature_cols)}")

    # NaN preserved — no fillna anywhere
    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = merged["y"]
    season = merged["season"]

    tr = season.isin(TRAIN_SEASONS)
    va = season == VAL_SEASON
    te = season == TEST_SEASON
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    log.info(f"split -> train {tr.sum():,} ({TRAIN_SEASONS})  "
             f"val {va.sum():,} ({VAL_SEASON})  test {te.sum():,} ({TEST_SEASON})")
    log.info(f"base rate -> train {ytr.mean():.3f}  val {yva.mean():.3f}  "
             f"test {yte.mean() if te.sum() else float('nan'):.3f}")

    results = {}
    for mt in ["random_forest", "histgbm", "xgboost"]:
        log.info(f"── optimizing {mt} ──")
        best = optuna_search(mt, Xtr, ytr, Xva, yva)
        model = fit_final(mt, best, Xtr, ytr)
        pv = model.predict_proba(Xva)[:, 1]
        row = {"params": best,
               "val_auc": roc_auc_score(yva, pv),
               "val_prauc": average_precision_score(yva, pv)}
        if te.sum():
            pt = model.predict_proba(Xte)[:, 1]
            row["test_auc"] = roc_auc_score(yte, pt)
            row["test_prauc"] = average_precision_score(yte, pt)
        results[mt] = (model, row)
        msg = f"{mt}: val AUC {row['val_auc']:.4f}  PR-AUC {row['val_prauc']:.4f}"
        if te.sum():
            msg += f"   test AUC {row['test_auc']:.4f}  PR-AUC {row['test_prauc']:.4f}"
        else:
            msg += "   (no 2025 labels in PGL — test slice empty)"
        log.info(msg)

    best_mt = max(results, key=lambda k: results[k][1]["val_auc"])
    best_model = results[best_mt][0]
    log.info(f"BEST on val: {best_mt}")

    # permutation importance on val, with td_prob watch
    log.info("── permutation importance (val) ──")
    Xva_imp = Xva.fillna(Xva.median())  # perm needs finite; median only for this diagnostic
    perm = permutation_importance(best_model, Xva_imp, yva, n_repeats=10,
                                  random_state=42, scoring="roc_auc", n_jobs=-1)
    imp = (pd.DataFrame({"feature": feature_cols, "imp": perm.importances_mean})
             .sort_values("imp", ascending=False).reset_index(drop=True))
    for _, r in imp.iterrows():
        tag = "  <-- nfl_data_py TD model" if r["feature"] in NFLDP_TDPROB_COLS else ""
        log.info(f"    {r['feature']:<28}{r['imp']:+.4f}{tag}")

    tdp = imp[imp["feature"].isin(NFLDP_TDPROB_COLS)]
    if len(tdp) and imp.iloc[0]["feature"] in NFLDP_TDPROB_COLS:
        log.warning("sum_td_prob DOMINATES permutation importance. You are partly "
                    "predicting nfl_data_py's own TD model. Re-run with "
                    "INCLUDE_NFLDP_TDPROB=False and compare AUC delta.")

    import joblib
    joblib.dump({"model": best_model, "model_type": best_mt,
                 "features": feature_cols, "results": {k: v[1] for k, v in results.items()}},
                args.model_out)
    log.info(f"saved -> {args.model_out}")


if __name__ == "__main__":
    main()