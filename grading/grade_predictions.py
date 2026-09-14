#!/usr/bin/env python3
"""
grade_predictions.py — grade NFL anytime-TD predictions against ESPN results
=============================================================================
Reads the exact week-specific prediction/parlay CSVs produced before kickoff by
live/predict_td.py. It never reconstructs Monday's betting decisions afterward.

Examples
--------
Repo:
    python grading/grade_predictions.py --season 2026 --week 1

Local Documents workflow:
    python grade_predictions.py --season 2026 --week 1 \
        --predictions predictions/nfl_td_predictions_2026_week01.csv \
        --parlays predictions/nfl_td_parlays_2026_week01.csv \
        --out-dir results
"""

from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
SEP = "=" * 112


def normalize_player_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("ASCII")
    name = re.sub(r"[^\w\s]", "", name).strip().lower()
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    return re.sub(r"\s+", " ", name).strip()


TEAM_ALIASES = {
    "WSH": "WAS", "JAC": "JAX", "LA": "LAR", "STL": "LAR",
    "SD": "LAC", "OAK": "LV",
}

def normalize_team(team) -> str:
    t = str(team or "").strip().upper()
    return TEAM_ALIASES.get(t, t)


def american_to_decimal(odds) -> float:
    a = float(odds)
    return a / 100.0 + 1.0 if a > 0 else 100.0 / abs(a) + 1.0


def get_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        if attempt + 1 < retries:
            time.sleep(1)
    return None


def get_week_games(season: int, week: int):
    data = get_json(
        f"{ESPN}/scoreboard",
        {"dates": season, "seasontype": 2, "week": week},
    )
    return (data or {}).get("events", [])


def _safe_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def fetch_game_actuals(game):
    game_id = str(game.get("id", ""))
    comp = (game.get("competitions") or [{}])[0]
    competitors = comp.get("competitors", [])
    home = normalize_team(next((c.get("team", {}).get("abbreviation") for c in competitors if c.get("homeAway") == "home"), ""))
    away = normalize_team(next((c.get("team", {}).get("abbreviation") for c in competitors if c.get("homeAway") == "away"), ""))
    completed = str((comp.get("status") or {}).get("type", {}).get("completed", False)).lower() == "true"
    if not completed:
        completed = bool((comp.get("status") or {}).get("type", {}).get("completed", False))

    data = get_json(f"{ESPN}/summary", {"event": game_id})
    if not data:
        return [], {"game_id": game_id, "home": home, "away": away, "completed": completed}

    # Aggregate every athlete appearing in the box score. A player may occur in
    # multiple stat categories; rushing + receiving TDs are summed separately.
    players = {}
    for team_block in (data.get("boxscore") or {}).get("players", []):
        team = normalize_team((team_block.get("team") or {}).get("abbreviation", ""))
        opp = away if team == home else home
        for category in team_block.get("statistics", []):
            cname = str(category.get("name", "")).lower()
            labels = category.get("labels", [])
            for athlete_stat in category.get("athletes", []):
                athlete = athlete_stat.get("athlete") or {}
                name = athlete.get("displayName") or athlete.get("fullName") or ""
                pid = _safe_int(athlete.get("id"))
                if not name:
                    continue
                key = (pid, normalize_player_name(name), team)
                rec = players.setdefault(key, {
                    "player_id": pid,
                    "player": name,
                    "team": team,
                    "opponent": opp,
                    "rushing_td": 0,
                    "receiving_td": 0,
                    "game_id": game_id,
                    "appeared_in_boxscore": True,
                })
                stat_dict = dict(zip(labels, athlete_stat.get("stats", [])))
                td = _safe_int(stat_dict.get("TD")) or 0
                if "rushing" in cname:
                    rec["rushing_td"] += td
                elif "receiving" in cname:
                    rec["receiving_td"] += td

    out = []
    for rec in players.values():
        rec["total_td"] = rec["rushing_td"] + rec["receiving_td"]
        out.append(rec)
    return out, {"game_id": game_id, "home": home, "away": away, "completed": completed}


def collect_actuals(season: int, week: int):
    games = get_week_games(season, week)
    if not games:
        raise RuntimeError(f"No ESPN games found for season={season}, week={week}")

    actual_rows = []
    game_meta = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_game_actuals, g): g.get("id") for g in games}
        for fut in as_completed(futures):
            rows, meta = fut.result()
            actual_rows.extend(rows)
            game_meta.append(meta)

    actuals = pd.DataFrame(actual_rows)
    if len(actuals):
        actuals["norm_name"] = actuals["player"].map(normalize_player_name)
    return actuals, pd.DataFrame(game_meta)


def _prediction_match(pred, actuals: pd.DataFrame):
    if actuals.empty:
        return None

    pid = _safe_int(pred.get("player_id"))
    team = normalize_team(pred.get("team"))
    if pid is not None and "player_id" in actuals.columns:
        m = actuals[actuals["player_id"] == pid]
        if team and len(m):
            mt = m[m["team"] == team]
            if len(mt):
                m = mt
        if len(m):
            return m.iloc[0]

    norm = normalize_player_name(pred.get("player", ""))
    m = actuals[actuals["norm_name"] == norm]
    if team and len(m):
        mt = m[m["team"] == team]
        if len(mt):
            m = mt
    return m.iloc[0] if len(m) else None


def grade_predictions(preds: pd.DataFrame, actuals: pd.DataFrame, games: pd.DataFrame):
    completed_teams = set()
    if len(games):
        done = games[games["completed"] == True]  # noqa: E712
        completed_teams.update(done["home"].dropna().map(normalize_team))
        completed_teams.update(done["away"].dropna().map(normalize_team))

    rows = []
    for _, p in preds.iterrows():
        a = _prediction_match(p, actuals)
        rec = p.to_dict()
        line = float(p.get("line", 0.5))

        if a is not None:
            actual_tds = int(a["total_td"])
            rec.update({
                "actual_tds": actual_tds,
                "rushing_td": int(a["rushing_td"]),
                "receiving_td": int(a["receiving_td"]),
                "result_status": "GRADED",
                "hit": bool(actual_tds > line),
            })
        elif normalize_team(p.get("team")) not in completed_teams:
            rec.update({
                "actual_tds": np.nan,
                "rushing_td": np.nan,
                "receiving_td": np.nan,
                "result_status": "GAME_NOT_FINAL",
                "hit": np.nan,
            })
        else:
            # ESPN did not list the athlete in any box-score category. Do not
            # silently turn this into a loss or DNP; flag it for review.
            rec.update({
                "actual_tds": np.nan,
                "rushing_td": np.nan,
                "receiving_td": np.nan,
                "result_status": "UNRESOLVED_NO_BOX_SCORE",
                "hit": np.nan,
            })

        # P/L uses the exact pregame recommended stake, not a reconstructed bet.
        stake = float(p.get("stake_units", 0) or 0)
        should_bet = str(p.get("should_bet", False)).lower() in {"true", "1"}
        odds = p.get("odds")
        if should_bet and stake > 0 and pd.notna(rec["hit"]) and pd.notna(odds):
            if bool(rec["hit"]):
                rec["pnl_units"] = stake * (american_to_decimal(odds) - 1.0)
            else:
                rec["pnl_units"] = -stake
        else:
            rec["pnl_units"] = 0.0
        rows.append(rec)

    return pd.DataFrame(rows)


def grade_parlays(parlays: pd.DataFrame, graded: pd.DataFrame):
    if parlays is None or parlays.empty:
        return pd.DataFrame()

    out = []
    for pid, grp in parlays.groupby("parlay_id", sort=True):
        leg_results = []
        unresolved = False
        all_hit = True
        for _, leg in grp.sort_values("leg_number").iterrows():
            player_id = _safe_int(leg.get("player_id"))
            candidates = graded
            if player_id is not None and "player_id" in graded.columns:
                m = graded[pd.to_numeric(graded["player_id"], errors="coerce") == player_id]
            else:
                m = graded[graded["player"].map(normalize_player_name) == normalize_player_name(leg.get("player", ""))]
            if len(m):
                r = m.iloc[0]
                hit = r.get("hit")
                status = r.get("result_status", "")
            else:
                hit, status = np.nan, "UNRESOLVED"

            if pd.isna(hit):
                unresolved = True
                all_hit = False
            elif not bool(hit):
                all_hit = False
            leg_results.append(f"{leg.get('player')}={'?' if pd.isna(hit) else ('HIT' if bool(hit) else 'MISS')}")

        first = grp.iloc[0]
        status = "UNRESOLVED" if unresolved else ("HIT" if all_hit else "MISS")
        # No parlay stake is recommended by the live script. Track a transparent
        # 1-unit hypothetical result for evaluation only.
        dec = float(first.get("parlay_dec_odds", np.nan))
        hypothetical_1u = np.nan
        if status == "HIT" and np.isfinite(dec):
            hypothetical_1u = dec - 1.0
        elif status == "MISS":
            hypothetical_1u = -1.0

        out.append({
            "parlay_id": pid,
            "parlay_type": first.get("parlay_type", ""),
            "num_legs": int(first.get("num_legs", len(grp))),
            "parlay_probability": first.get("parlay_probability"),
            "parlay_dec_odds": first.get("parlay_dec_odds"),
            "parlay_ev": first.get("parlay_ev"),
            "result": status,
            "hypothetical_1u_pnl": hypothetical_1u,
            "legs": " | ".join(leg_results),
        })
    return pd.DataFrame(out)


def print_summary(graded: pd.DataFrame, parlay_results: pd.DataFrame, season: int, week: int):
    print(f"\n{SEP}\nNFL ANYTIME-TD RESULTS — {season} WEEK {week}\n{SEP}")

    resolved = graded[graded["result_status"] == "GRADED"].copy()
    if len(resolved):
        print(f"All projections: {int(resolved['hit'].sum())}/{len(resolved)} TD hits "
              f"({resolved['hit'].mean():.1%})")
    else:
        print("All projections: no resolved rows")

    bets = resolved[resolved["should_bet"].astype(str).str.lower().isin(["true", "1"])].copy()
    if len(bets):
        risked = pd.to_numeric(bets["stake_units"], errors="coerce").fillna(0).sum()
        pnl = pd.to_numeric(bets["pnl_units"], errors="coerce").fillna(0).sum()
        roi = pnl / risked if risked > 0 else np.nan
        print(f"Qualified singles: {int(bets['hit'].sum())}/{len(bets)} "
              f"({bets['hit'].mean():.1%}) | risked {risked:.2f}u | "
              f"P/L {pnl:+.2f}u | ROI {roi:+.1%}" if np.isfinite(roi) else
              f"Qualified singles: {len(bets)} | P/L {pnl:+.2f}u")
    else:
        print("Qualified singles: none")

    unresolved = graded[graded["result_status"] != "GRADED"]
    if len(unresolved):
        print(f"Unresolved rows: {len(unresolved)} (not counted as wins/losses or P/L)")

    if parlay_results is not None and len(parlay_results):
        h = (parlay_results["result"] == "HIT").sum()
        m = (parlay_results["result"] == "MISS").sum()
        u = (parlay_results["result"] == "UNRESOLVED").sum()
        print(f"Parlays: {h} hit / {m} miss / {u} unresolved "
              "(1u hypothetical tracking; live script does not recommend a parlay stake)")

    if len(resolved):
        print("\nProbability buckets:")
        bins = [0, .20, .30, .40, .50, .60, 1.01]
        labels = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60%+"]
        resolved["bucket"] = pd.cut(resolved["p_td"], bins=bins, labels=labels, right=False)
        tab = resolved.groupby("bucket", observed=True)["hit"].agg(["count", "sum", "mean"])
        for bucket, row in tab.iterrows():
            print(f"  {str(bucket):>7}: {int(row['sum']):>2}/{int(row['count']):<2}  {row['mean']:.1%}")


def default_paths(season: int, week: int):
    here = Path(__file__).resolve()
    project_root = here.parents[1] if here.parent.name == "grading" else here.parent
    pred = project_root / "predictions" / f"nfl_td_predictions_{season}_week{week:02d}.csv"
    parlays = project_root / "predictions" / f"nfl_td_parlays_{season}_week{week:02d}.csv"
    out_dir = project_root / "results"
    return pred, parlays, out_dir


def main():
    ap = argparse.ArgumentParser(description="Grade saved NFL anytime-TD predictions")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--predictions", type=str)
    ap.add_argument("--parlays", type=str)
    ap.add_argument("--out-dir", type=str)
    args = ap.parse_args()

    d_pred, d_parlay, d_out = default_paths(args.season, args.week)
    pred_path = Path(os.path.expanduser(args.predictions)) if args.predictions else d_pred
    parlay_path = Path(os.path.expanduser(args.parlays)) if args.parlays else d_parlay
    out_dir = Path(os.path.expanduser(args.out_dir)) if args.out_dir else d_out

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction snapshot not found: {pred_path}")

    preds = pd.read_csv(pred_path)
    required = {"player", "team", "p_td", "season", "week", "should_bet", "stake_units"}
    missing = sorted(required - set(preds.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    if not (pd.to_numeric(preds["season"], errors="coerce") == args.season).all():
        raise ValueError("Prediction CSV season does not match --season")
    if not (pd.to_numeric(preds["week"], errors="coerce") == args.week).all():
        raise ValueError("Prediction CSV week does not match --week")

    parlays = pd.read_csv(parlay_path) if parlay_path.exists() else pd.DataFrame()

    print(f"Loading predictions: {pred_path}")
    print(f"Fetching ESPN actuals for {args.season} Week {args.week}...")
    actuals, games = collect_actuals(args.season, args.week)
    graded = grade_predictions(preds, actuals, games)
    parlay_results = grade_parlays(parlays, graded)

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"nfl_td_results_{args.season}_week{args.week:02d}.csv"
    graded.to_csv(result_path, index=False)

    parlay_result_path = out_dir / f"nfl_td_parlay_results_{args.season}_week{args.week:02d}.csv"
    parlay_results.to_csv(parlay_result_path, index=False)

    print_summary(graded, parlay_results, args.season, args.week)
    print(f"\nSaved prediction results -> {result_path}")
    print(f"Saved parlay results     -> {parlay_result_path}")


if __name__ == "__main__":
    main()
