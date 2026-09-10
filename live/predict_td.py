# %%
#!/usr/bin/env python3
"""
predict_td.py — NFL anytime-TD live predictions
==================================================================
Full rewrite of the 5,947-line CSV-driven live script. The new model
(nfl_td_model_v2.joblib) runs on the same 30 leak-free features the trainer
built. Historical model features come from the DB/nflverse pipeline, while
the upcoming slate and current player identity (team/position) come from ESPN.
The six weekly-updated red-zone CSVs and backup-QB probability machinery used
by the old feature set are gone. Sleeper injury/status data is retained only as
an availability filter so confirmed non-playing players never reach the final
prediction tables.

WHAT IT DOES
  1. pull the week's slate from ESPN
  2. rebuild each active player's 30 features AS-OF the upcoming week (rolled
     through the prior week — identical to NFL_TD_REAL_PREDICTIONS.py, which
     matched real players correctly)
  3. score P(anytime TD) with the saved model
  4. fetch book anytime-TD odds (The Odds API) and flag value where the model's
     probability beats the book's implied probability
  5. tag low-confidence rows (rookies / no prior history) explicitly

WEEK 1 / PRESEASON FALLBACK
  Before the season starts, nfl_data_py has no 2026 games. The feature builder
  falls back to each player's MOST RECENT completed season so Week 1 isn't
  blank. A player with NO history in any loaded season (true rookie) gets NaN
  features -> the model uses its learned "unknown" branch, and the row is
  tagged LOW-CONF so you know the projection is unreliable. Seeding rookies
  from preseason touch projections is a future upgrade (see ROOKIE_PROJECTIONS).

ROOKIE HANDLING (current)
  A rookie bell-cow (think a Week-1 workhorse with no NFL snaps) is the one
  blind spot: NaN rolling features -> the model regresses him to the mean,
  which UNDER-rates a true lead back. Rows are tagged so you don't bet them
  blind. If you have a CSV of preseason projected touches, point
  ROOKIE_PROJECTIONS at it and the builder seeds roll*_touch from it.
"""

import os
import time
import warnings
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DB = Path(os.getenv(
    "NFL_TD_DB",
    str(PROJECT_ROOT / "data" / "nfl_props_COMPLETE.db"),
))
BUNDLE = Path(os.getenv(
    "NFL_TD_MODEL",
    str(PROJECT_ROOT / "models" / "nfl_td_model_v2.joblib"),
))
CROSSWALK = Path(os.getenv(
    "NFL_TD_CROSSWALK",
    str(PROJECT_ROOT / "data" / "nfl_id_crosswalk.csv"),
))
ROOKIE_PROJECTIONS = Path(os.getenv(
    "NFL_TD_ROOKIE_PROJECTIONS",
    str(PROJECT_ROOT / "data" / "rookie_touch_projections.csv"),
))

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
NFL_SPORT = "americanfootball_nfl"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

CURRENT_SEASON = 2026          # the season we're predicting
FALLBACK_SEASONS = [2025, 2024]  # used when CURRENT_SEASON has no games yet
VALUE_EDGE = 0.03              # model prob must beat book implied by this to flag
FORCE_WEEK = None             # None = ESPN's current/upcoming week; set an int to force

# Disk cache: survives notebook/script restarts and reduces repeat API calls.
CACHE_DIR = Path(os.getenv("NFL_TD_CACHE_DIR", str(PROJECT_ROOT / "cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL = {
    "espn_slate": 30 * 60,       # 30 min
    "espn_roster": 6 * 60 * 60,  # 6 hr
    "sleeper_players": 30 * 60,  # 30 min
    "odds_events": 10 * 60,      # 10 min
    "odds_props": 10 * 60,       # 10 min
}

PBP_ROLL_COLS = [
    "targets", "ez_targets", "inside10_targets", "inside5_targets",
    "air_yards_game", "adot_game", "sum_td_prob", "ez_target_share",
    "carries", "ez_carries", "inside5_carries", "rush_sum_td_prob",
    "ngs_separation", "ngs_cushion", "ngs_intended_air", "ngs_pct_air_share",
    "offense_pct"]
import re


NFL_TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}


def normalize_team_name(team: str) -> str:
    """Normalize team names to standard abbreviations using existing NFL_TEAMS dict"""
    
    # Create reverse mapping from full name to abbreviation
    FULL_TO_ABBREV = {v: k for k, v in NFL_TEAMS.items()}
    
    # Handle alternate abbreviations
    abbrev_mapping = {
        'SFO': 'SF',
        'NWE': 'NE',
        'KAN': 'KC',
        'TAM': 'TB',
        'GNB': 'GB',
        'LVR': 'LV',
        'WSH': 'WAS'
    }
    
    team = team.strip()
    
    # If it's an alternate abbreviation, standardize it
    if team in abbrev_mapping:
        return abbrev_mapping[team]
    
    # If it's already a standard abbreviation, return it
    if team in NFL_TEAMS:
        return team
    
    # If it's a full name, convert to abbreviation
    return FULL_TO_ABBREV.get(team, team)


def normalize_player_name(name: str) -> str:
    """Normalize player names for matching - FIXED VERSION"""
    if not name:
        return ""
    
    # Remove common punctuation and standardize
    name = re.sub(r'[^\w\s]', '', name).strip().lower()
    
    # FIXED: Remove suffixes with spaces - this was the bug!
    name = re.sub(r'\s+(jr|sr|ii|iii|iv)$', '', name)  # FIXED LINE
    
    # Handle middle initials - remove them for better matching
    name = re.sub(r'\s+[a-z]\.\s+', ' ', name)
    name = re.sub(r'\s+[a-z]\s+', ' ', name)
    
    # Normalize multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


SEP = "=" * 90


def _cache_path(namespace, url, params):
    payload = json.dumps(
        {"url": url, "params": params or {}},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return CACHE_DIR / f"{namespace}_{digest}.json"


def _read_cache(namespace, url, params, ttl):
    path = _cache_path(namespace, url, params)
    if not path.exists():
        return None

    age = time.time() - path.stat().st_mtime
    if age > ttl:
        try:
            path.unlink()
        except OSError:
            pass
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _write_cache(namespace, url, params, data):
    path = _cache_path(namespace, url, params)
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def get(url, params=None, retries=3, cache_namespace=None, cache_ttl=None,
        allow_stale_on_error=True):
    """
    GET JSON with optional disk caching.

    cache_namespace/cache_ttl omitted -> uncached request.
    If a live request fails and a stale cache exists, optionally use it as a
    fallback so a temporary API outage does not kill the whole slate.
    """
    if cache_namespace and cache_ttl:
        cached = _read_cache(cache_namespace, url, params, cache_ttl)
        if cached is not None:
            return cached

    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if cache_namespace and cache_ttl:
                    _write_cache(cache_namespace, url, params, data)
                return data
        except Exception:
            pass
        time.sleep(1)

    if cache_namespace and allow_stale_on_error:
        path = _cache_path(cache_namespace, url, params)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    print(f"  using stale {cache_namespace} cache after API failure")
                    return json.load(f)
            except Exception:
                pass

    return None


def american_to_prob(american):
    a = float(american)
    return (-a) / (-a + 100) if a < 0 else 100 / (a + 100)


def prob_to_american(p):
    p = min(max(p, 1e-4), 0.9999)
    return round(-100 * p / (1 - p)) if p > 0.5 else round(100 * (1 - p) / p)


# ── SLATE RESOLUTION (which week, which teams play) ─────────────────────────
def resolve_slate(week=None):
    """Return (season, week, playing_team_abbrs) for the upcoming slate.

    ESPN's scoreboard auto-advances: hit with no week and it returns the
    CURRENT/UPCOMING week's games even if today has none (NFL is weekly, so
    'today' is usually empty). Pass week= to force a specific week.
    """
    params = {}
    if week:
        params["week"] = week
    data = get(
        f"{ESPN}/scoreboard",
        params,
        cache_namespace="espn_slate",
        cache_ttl=CACHE_TTL["espn_slate"],
    )
    if not data:
        return None, None, set()
    wk = (data.get("week") or {}).get("number", week or 1)
    seas = (data.get("season") or {}).get("year", CURRENT_SEASON)
    team_ids = set()
    matchup = {}      # team_id -> {"abbr":, "opp":} for labelling the table
    for ev in data.get("events", []):
        for comp in ev.get("competitions", []):
            cs = comp.get("competitors", [])
            ids = []
            for c in cs:
                tid = c.get("team", {}).get("id")
                ab = c.get("team", {}).get("abbreviation")
                if tid is not None:
                    try:
                        tid = int(tid); team_ids.add(tid)
                        ids.append((tid, ab))
                    except (TypeError, ValueError):
                        pass
            # two competitors -> each is the other's opponent
            if len(ids) == 2:
                (t0, a0), (t1, a1) = ids
                matchup[t0] = {"abbr": a0, "opp": a1}
                matchup[t1] = {"abbr": a1, "opp": a0}
    return seas, wk, team_ids, matchup


# ── FEATURE BUILD (identical to the trainer, minus the label) ────────────────
def roll_asof(df, cols, group="player_id", prefix="r_"):
    out = {}
    for c in cols:
        if c in df.columns:
            out[prefix + c] = df.groupby(group)[c].transform(
                lambda s: s.expanding().mean().shift(1))
    return pd.DataFrame(out, index=df.index)


def build_history(conn):
    """All player-games from the fallback seasons, with rolled features, so the
    LAST row per player is his current as-of form for the upcoming game."""
    seasons = ",".join(str(s) for s in [CURRENT_SEASON] + FALLBACK_SEASONS)
    base = pd.read_sql(f"""
        SELECT p.player_id, p.player_name, p.season, p.week, p.opponent_id,
               p.team_id,
               COALESCE(p.rushing_tds,0)+COALESCE(p.receiving_tds,0) AS total_tds,
               COALESCE(p.rushing_attempts,0)+COALESCE(p.targets,0) AS total_touches,
               g.temperature, g.dome_game, g.weather_impact_score,
               g.point_spread, g.over_under
        FROM player_game_logs p JOIN game_context g ON p.game_id=g.game_id
        WHERE p.season IN ({seasons}) AND (p.rushing_attempts>=1 OR p.targets>=1)
        ORDER BY p.player_id, p.season, p.week""", conn)
    pbp = pd.read_sql("SELECT * FROM pbp_td_features", conn)
    pbp["week"] = pd.to_numeric(pbp["game_id"].astype(str).str.split("_").str[1],
                                errors="coerce")
    additive = ["targets", "ez_targets", "inside10_targets", "inside5_targets",
                "air_yards_game", "sum_td_prob", "carries", "ez_carries",
                "inside5_carries", "rush_sum_td_prob"]
    rate = ["adot_game", "ez_target_share", "ngs_separation", "ngs_cushion",
            "ngs_intended_air", "ngs_pct_air_share", "offense_pct"]
    agg = {c: "sum" for c in additive if c in pbp.columns}
    agg.update({c: "max" for c in rate if c in pbp.columns})
    pbp = pbp.groupby(["player_id", "season", "week"], as_index=False).agg(agg)

    m = base.merge(pbp, on=["player_id", "season", "week"], how="left")
    m = m.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    m = pd.concat([m, roll_asof(m, PBP_ROLL_COLS)], axis=1)
    for w in (3, 5, 8):
        m[f"roll{w}_tds"] = m.groupby("player_id")["total_tds"].transform(
            lambda s: s.rolling(w, min_periods=1).mean().shift(1))
        m[f"roll{w}_touch"] = m.groupby("player_id")["total_touches"].transform(
            lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    m["td_momentum"] = m.groupby("player_id")["total_tds"].transform(
        lambda s: s.rolling(3, min_periods=1).sum().shift(1))
    allowed = (m.groupby(["opponent_id", "season", "week"], as_index=False)["total_tds"]
               .sum().rename(columns={"opponent_id": "def_team",
                                      "total_tds": "tds_allowed"})
               .sort_values(["def_team", "season", "week"]))
    allowed["def_tds_allowed_asof"] = allowed.groupby(
        ["def_team", "season"])["tds_allowed"].transform(
        lambda s: s.expanding().mean().shift(1))
    m = m.merge(allowed[["def_team", "season", "week", "def_tds_allowed_asof"]],
                left_on=["opponent_id", "season", "week"],
                right_on=["def_team", "season", "week"], how="left")
    return m


def current_form(hist):
    """One row per player = his most recent game's as-of features. This is what
    the upcoming game is predicted from. Note whether it came from CURRENT_SEASON
    or a fallback, so Week 1 can be flagged."""
    latest = (hist.dropna(subset=["roll8_touch"])
              .sort_values(["player_id", "season", "week"])
              .groupby("player_id").tail(1).copy())
    latest["from_current_season"] = latest["season"] == CURRENT_SEASON
    return latest


def seed_rookies(latest, model_feats):
    """Optionally seed rookies (no history) from preseason touch projections."""
    if not os.path.exists(ROOKIE_PROJECTIONS):
        return latest
    proj = pd.read_csv(ROOKIE_PROJECTIONS)   # player_name, proj_touches, proj_ez
    have = set(latest["player_name"])
    add = proj[~proj["player_name"].isin(have)].copy()
    if not len(add):
        return latest
    for w in (3, 5, 8):
        add[f"roll{w}_touch"] = add["proj_touches"]
    add["r_ez_carries"] = add.get("proj_ez", np.nan)
    add["from_current_season"] = False
    add["is_rookie_seed"] = True
    print(f"  seeded {len(add)} rookies from {os.path.basename(ROOKIE_PROJECTIONS)}")
    return pd.concat([latest, add], ignore_index=True)



# ── CURRENT ROSTERS ───────────────────────────────────────────────────────────
def fetch_current_rosters(playing_team_ids, matchup):
    """
    Current-season identity map from ESPN rosters.

    Historical DB rows provide form/features only. Team and position for the
    upcoming game come from the CURRENT_SEASON ESPN roster so offseason trades,
    free-agent moves and rookies do not inherit stale historical teams.

    Returns
    -------
    by_id:
        ESPN player_id -> current roster record
    by_name:
        normalized player name -> current roster record
    """
    by_id = {}
    by_name = {}

    team_ids = sorted(int(t) for t in playing_team_ids) if playing_team_ids else []
    if not team_ids:
        # If slate resolution failed, use the ESPN team IDs we already know.
        team_ids = sorted(matchup.keys())

    for team_id in team_ids:
        mu = matchup.get(team_id, {})
        team_abbr = mu.get("abbr", "")
        opp_abbr = mu.get("opp", "")

        data = get(
            f"{ESPN}/teams/{team_id}/roster",
            cache_namespace="espn_roster",
            cache_ttl=CACHE_TTL["espn_roster"],
        )
        if not data:
            print(f"  roster unavailable: {team_abbr or team_id}")
            continue

        for group in data.get("athletes", []):
            for athlete in group.get("items", []):
                pid = athlete.get("id")
                name = athlete.get("displayName") or athlete.get("fullName") or ""
                pos = (athlete.get("position") or {}).get("abbreviation", "")
                status = (athlete.get("status") or {}).get("type", "")

                if not pid or not name:
                    continue

                try:
                    pid = int(pid)
                except (TypeError, ValueError):
                    continue

                rec = {
                    "player_id": pid,
                    "player_name": name,
                    "team_id": int(team_id),
                    "team": team_abbr,
                    "opp": opp_abbr,
                    "pos": pos,
                    "status": status,
                }
                by_id[pid] = rec

                norm = normalize_player_name(name)
                if norm:
                    by_name[norm] = rec

    return by_id, by_name


def attach_current_roster(latest, playing_team_ids, matchup):
    """
    Attach current team/position without altering historical form columns.

    Existing players match by ESPN player_id. Rookie projection rows that do
    not yet carry an ESPN ID can match by normalized player name.
    """
    by_id, by_name = fetch_current_rosters(playing_team_ids, matchup)

    if not by_id:
        print("  WARNING: no current ESPN roster data returned")
        return latest.iloc[0:0].copy(), by_id

    out = latest.copy()

    def roster_rec(row):
        pid = row.get("player_id")
        if pd.notna(pid):
            try:
                rec = by_id.get(int(pid))
                if rec:
                    return rec
            except (TypeError, ValueError):
                pass

        return by_name.get(normalize_player_name(str(row.get("player_name", ""))))

    recs = out.apply(roster_rec, axis=1)
    matched = recs.notna()

    before = len(out)
    out = out.loc[matched].copy()
    recs = recs.loc[matched]

    out["current_team_id"] = [r["team_id"] for r in recs]
    out["current_team"] = [r["team"] for r in recs]
    out["current_opp"] = [r["opp"] for r in recs]
    out["current_pos"] = [r["pos"] for r in recs]
    out["roster_status"] = [r["status"] for r in recs]

    # Keep only teams on the resolved slate. This check uses CURRENT roster
    # identity, never the player's last historical team.
    if playing_team_ids:
        out = out[out["current_team_id"].isin(set(playing_team_ids))].copy()

    print(
        f"  current ESPN rosters: {len(by_id)} players | "
        f"{len(out)} model-history/rookie rows matched (from {before})"
    )

    # Helpful audit: show historical-team moves rather than silently overwriting.
    if "team_id" in out.columns:
        moved = out[
            out["team_id"].notna()
            & (pd.to_numeric(out["team_id"], errors="coerce")
               != pd.to_numeric(out["current_team_id"], errors="coerce"))
        ]
        if len(moved):
            print(f"  current-team overrides: {len(moved)} players moved since their form source")
            for _, r in moved.head(12).iterrows():
                old_id = r.get("team_id")
                print(
                    f"    {r['player_name']}: historical team_id {old_id} -> "
                    f"{r['current_team']} ({int(r['current_team_id'])})"
                )
            if len(moved) > 12:
                print(f"    ... plus {len(moved) - 12} more")

    return out, by_id



# ── INJURY / AVAILABILITY FILTER ──────────────────────────────────────────────
# Sleeper's player map exposes current status, injury_status, practice status,
# injury body part and team. We use it only as an availability guard; it does
# NOT alter model probabilities.
SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"

# Confirmed / near-confirmed non-playing statuses to exclude entirely.
# Questionable is intentionally NOT excluded because those players can play.
EXCLUDE_INJURY_STATUSES = {
    "out",
    "ir",
    "injured reserve",
    "pup",
    "physically unable to perform",
    "nfi",
    "non-football injury",
    "inactive",
    "suspended",
    "reserve",
    "doubtful",
}

EXCLUDE_GENERAL_STATUSES = {
    "inactive",
    "injured reserve",
    "reserve",
    "suspended",
    "pup",
    "nfi",
}


def fetch_sleeper_availability():
    """
    Return normalized-name -> list of Sleeper player records.

    One call covers the slate. Matching is done after current ESPN team identity
    is attached, so a player's historical team never controls injury filtering.
    """
    data = get(
        SLEEPER_PLAYERS,
        cache_namespace="sleeper_players",
        cache_ttl=CACHE_TTL["sleeper_players"],
    )
    if not isinstance(data, dict):
        print("  WARNING: Sleeper injury data unavailable — no injury filter applied")
        return {}

    by_name = {}
    for _, p in data.items():
        name = p.get("full_name")
        if not name:
            first = p.get("first_name") or ""
            last = p.get("last_name") or ""
            name = f"{first} {last}".strip()
        if not name:
            continue

        norm = normalize_player_name(name)
        if not norm:
            continue

        rec = {
            "name": name,
            "team": normalize_team_name(str(p.get("team") or "")),
            "position": str(p.get("position") or ""),
            "status": str(p.get("status") or ""),
            "injury_status": str(p.get("injury_status") or ""),
            "injury_body_part": str(p.get("injury_body_part") or ""),
            "practice_participation": str(p.get("practice_participation") or ""),
            "active": p.get("active"),
        }
        by_name.setdefault(norm, []).append(rec)

    return by_name


def _availability_record(player_name, current_team, sleeper_by_name):
    candidates = sleeper_by_name.get(normalize_player_name(player_name), [])
    if not candidates:
        return None

    team = normalize_team_name(str(current_team or ""))
    team_matches = [p for p in candidates if p.get("team") == team]

    if len(team_matches) == 1:
        return team_matches[0]
    if len(team_matches) > 1:
        return team_matches[0]

    # Name-only fallback only when unambiguous. This avoids filtering the wrong
    # player when duplicate names exist.
    if len(candidates) == 1:
        return candidates[0]

    return None


def _is_confirmed_out(av):
    if not av:
        return False, ""

    injury = str(av.get("injury_status") or "").strip().lower()
    status = str(av.get("status") or "").strip().lower()
    active = av.get("active")

    reason = injury or status

    if injury in EXCLUDE_INJURY_STATUSES:
        return True, reason
    if status in EXCLUDE_GENERAL_STATUSES:
        return True, reason
    if active is False and (injury or status):
        return True, reason

    return False, reason


def filter_unavailable_players(latest):
    """
    Remove confirmed non-playing players BEFORE scoring output / odds decisions.

    Questionable players remain in the pool. OUT/IR/PUP/NFI/Suspended/Inactive
    and Doubtful players are removed whether or not a sportsbook has posted odds.
    """
    sleeper = fetch_sleeper_availability()
    if not sleeper or latest.empty:
        return latest

    keep_idx = []
    removed = []

    for idx, row in latest.iterrows():
        av = _availability_record(
            row.get("player_name", ""),
            row.get("current_team", ""),
            sleeper,
        )
        is_out, reason = _is_confirmed_out(av)

        if is_out:
            removed.append({
                "player": row.get("player_name", ""),
                "team": row.get("current_team", ""),
                "reason": reason or "inactive",
                "body_part": (av or {}).get("injury_body_part", ""),
            })
        else:
            keep_idx.append(idx)

    out = latest.loc[keep_idx].copy()

    if removed:
        print(f"  injury filter: removed {len(removed)} confirmed unavailable players")
        for r in removed[:15]:
            detail = f" ({r['body_part']})" if r["body_part"] else ""
            print(f"    {r['player']} ({r['team']}): {r['reason']}{detail}")
        if len(removed) > 15:
            print(f"    ... plus {len(removed) - 15} more")
    else:
        print("  injury filter: no confirmed unavailable players removed")

    return out


# ── ODDS ─────────────────────────────────────────────────────────────────────
def fetch_td_odds():
    """Anytime-TD odds per player from The Odds API, best (highest) price."""
    if not ODDS_API_KEY:
        print("  ODDS_API_KEY is not set — continuing without sportsbook prices")
        return {}

    ev = get(
        f"https://api.the-odds-api.com/v4/sports/{NFL_SPORT}/events",
        {"apiKey": ODDS_API_KEY, "regions": "us"},
        cache_namespace="odds_events",
        cache_ttl=CACHE_TTL["odds_events"],
    )
    best = {}
    for e in (ev or []):
        od = get(
            f"https://api.the-odds-api.com/v4/sports/{NFL_SPORT}/events/{e['id']}/odds",
            {"apiKey": ODDS_API_KEY, "regions": "us",
             "markets": "player_anytime_td", "oddsFormat": "american"},
            cache_namespace="odds_props",
            cache_ttl=CACHE_TTL["odds_props"],
        )
        for bk in (od or {}).get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "player_anytime_td":
                    continue
                for oc in mk.get("outcomes", []):
                    nm, price = oc.get("description"), oc.get("price")
                    if nm is None or price is None:
                        continue
                    if nm not in best or price > best[nm]["price"]:
                        best[nm] = {"price": price, "book": bk.get("title")}
    return best


# ── BET SIZING (ported from should_bet_td_prop_v3: 1/8 Kelly, tiered edges) ──
def evaluate_bet(prob, odds, min_edge=0.05, min_prob=0.15):
    """Kelly-sized bet decision. prob is the MODEL's calibrated probability
    (the trainer already calibrated it — no extra calibration layer needed).
    Tiered edge requirements and 1/8 Kelly with a 5% cap, verbatim from the
    old should_bet_td_prop_v3 so staking behaviour is unchanged."""
    imp = american_to_prob(odds)
    edge = (prob - imp) / imp if imp > 0 else 0.0
    out = {"should_bet": False, "prob": prob, "implied": imp, "edge": edge,
           "stake_units": 0.0, "confidence": "LOW", "reason": ""}
    if edge < min_edge:
        out["reason"] = f"edge {edge:.1%} < {min_edge:.0%}"; return out
    if prob < min_prob:
        out["reason"] = f"prob {prob:.1%} < {min_prob:.0%}"; return out
    # tiered edge floors by probability band
    if prob >= 0.58:
        req, conf = 0.10, "MEDIUM"
    elif prob >= 0.50:
        req, conf = 0.06, "HIGH"
    elif prob >= 0.35:
        req, conf = 0.05, "MEDIUM"
    else:
        req, conf = 0.08, "LOW"
    if edge < req:
        out["reason"] = f"{conf} band needs {req:.0%} edge (has {edge:.1%})"; return out
    out["should_bet"], out["confidence"] = True, conf
    # 1/8 Kelly, confidence-adjusted, capped at 5% of bankroll
    dec = (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)
    b = dec - 1
    kelly = (b * prob - (1 - prob)) / b if b > 0 else 0
    frac = max(0, kelly * 0.125)
    mult = {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.50}[conf]
    out["stake_units"] = round(min(frac * mult, 0.05) * 100, 2)   # units, 1u=1% bankroll base
    return out


# ── PARLAYS (rebuilt to match the old generate_td_parlays: 0.75 haircut,
#    three types, ANYTIME-TD line only — the model has no 1.5 line) ──────────
PARLAY_HAIRCUT = 0.75   # verbatim from the old script: multiply joint prob DOWN
                        # to stay conservative. NOT a correlation boost — there
                        # was never one, and same-team TDs may anti-correlate.

def _parlay(legs, haircut=PARLAY_HAIRCUT):
    p = 1.0
    for c in legs:
        p *= c["p_td"]
    p *= haircut
    dec = 1.0
    for c in legs:
        o = c["odds"]
        dec *= (o / 100 + 1) if o > 0 else (100 / abs(o) + 1)
    ev = p * (dec - 1) - (1 - p)
    return {"legs": [dict(c) for c in legs], "num_legs": len(legs),
            "probability": p, "dec_odds": dec, "ev": ev}


def generate_parlays(bets):
    """Three parlay types, exactly as the old script produced them, on the
    ANYTIME-TD line (the only line the model outputs):
      1. straight probability parlay — best legs by model prob, up to 8
      2. HIGH-PROBABILITY 6-leg — highest joint probability among top-pool combos
      3. SLEEPER VALUE 6-leg — highest-EV 6 among top-pool combos
    No invented correlation number; the 0.75 haircut is the old behaviour."""
    import itertools
    pool = [b for b in bets if b["p_td"] >= 0.25 and b["odds"] is not None]
    # one leg per player already (bets is per-player)
    pool.sort(key=lambda b: -b["p_td"])
    out = []

    # 1. straight probability parlay (up to 8 legs, best prob)
    if len(pool) >= 2:
        legs = pool[:8]
        pl = _parlay(legs)
        pl["type"] = f"{len(legs)}-leg anytime-TD (probability)"
        out.append(pl)

    # 2 & 3. Exhaustive 6-leg combinations from the top ~10 by probability
    cand = pool[:10]
    if len(cand) >= 6:
        combos = list(itertools.combinations(cand, 6))
        scored = [_parlay(list(c)) for c in combos]
        # HIGH-PROB: max joint probability
        hp = max(scored, key=lambda x: x["probability"])
        hp = dict(hp); hp["type"] = "6-leg anytime-TD High-Probability"
        out.append(hp)
        # SLEEPER: max EV (tends to longer odds)
        sv = max(scored, key=lambda x: x["ev"])
        sv = dict(sv); sv["type"] = "6-leg anytime-TD Sleeper Value"
        if sv["legs"] != hp["legs"]:
            out.append(sv)
    return out



def _fmt_odds(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return f"{int(v):+d}"
    except Exception:
        return "—"


def _fmt_edge(row):
    e = (row.get("eval") or {}).get("edge")
    return "—" if e is None else f"{e:+.1%}"


def _form_label(row):
    rookie = row.get("is_rookie_seed", False)
    if pd.notna(rookie) and bool(rookie):
        return "ROOKIE"
    if not bool(row.get("low_conf", False)):
        return "2026"
    return "2025"


def print_prediction_table(rows, top_n=30):
    """Old-style ranked table adapted to the current 2026 model output."""
    if not rows:
        print(f"\n{SEP}\n  TOP PROJECTIONS\n{SEP}")
        print("  no valid projections")
        return

    top = sorted(rows, key=lambda r: r["p_td"], reverse=True)[:top_n]

    print(f"\n{SEP}\n  ANYTIME TD PROJECTIONS\n{SEP}")
    print(
        f"  {'#':>2} {'player':<24} {'pos':<4} {'tm':<4} {'matchup':<10}"
        f"{'P(TD)':>8} {'fair':>7} {'book':<14} {'odds':>7} {'edge':>8}"
        f" {'touch8':>7} {'ezC':>6} {'ezT':>6} {'form':>8}"
    )
    print("  " + "-" * 122)

    for i, r in enumerate(top, 1):
        team = str(r.get("team", ""))[:4]
        opp = str(r.get("opp", ""))[:4]
        matchup = f"{team}-{opp}" if team and opp else team or "—"
        fair = r.get("fair")
        fair_s = "—" if fair is None or pd.isna(fair) else f"{int(fair):+d}"
        book = str(r.get("book", "—") or "—")[:14]
        touch8 = r.get("touch8", np.nan)
        ezc = r.get("ez_car", np.nan)
        ezt = r.get("ez_tgt", np.nan)

        print(
            f"  {i:>2}. {str(r['player'])[:23]:<24} "
            f"{str(r.get('pos',''))[:3]:<4} {team:<4} {matchup:<10}"
            f"{r['p_td']:>8.1%} {fair_s:>7} {book:<14} {_fmt_odds(r.get('odds')):>7}"
            f"{_fmt_edge(r):>8} "
            f"{touch8:>7.1f} {ezc:>6.2f} {ezt:>6.2f} {_form_label(r):>8}"
        )

    print("\n  form: 2026=current-season history | 2025=prior-season fallback | ROOKIE=projection seed")


def print_single_bets(bets):
    print(f"\n{SEP}\n  QUALIFIED SINGLE BETS  (1/8 Kelly, tiered edges)\n{SEP}")
    if not bets:
        print("  no singles clear the edge thresholds this slate")
        return

    print(
        f"  {'player':<24}{'pos':>5}{'tm':>5}{'matchup':>10}"
        f"{'P(TD)':>8}{'odds':>8}{'edge':>8}{'conf':>9}{'units':>8}  book"
    )
    print("  " + "-" * 100)

    for r in bets:
        e = r["eval"]
        matchup = f"{r.get('team','')}-{r.get('opp','')}"
        print(
            f"  {r['player'][:23]:<24}{str(r.get('pos',''))[:3]:>5}"
            f"{str(r.get('team',''))[:4]:>5}{matchup[:9]:>10}"
            f"{r['p_td']:>8.1%}{_fmt_odds(r.get('odds')):>8}"
            f"{e['edge']:>+8.1%}{e['confidence']:>9}{e['stake_units']:>8.2f}"
            f"  {r.get('book','')}"
        )


def print_parlay_table(parlays):
    """Restore the old parlay table layout without changing current parlay math."""
    print(f"\n{SEP}\n  TD PARLAYS  (anytime-TD line, 0.75 haircut)\n{SEP}")

    if not parlays:
        print("  need >=2 qualified legs (P>=0.25, priced) to build parlays")
        return

    for i, p in enumerate(parlays, 1):
        am = prob_to_american(p["probability"])
        legs = p["legs"]
        avg_prob = np.mean([x["p_td"] for x in legs]) if legs else np.nan
        edges = [(x.get("eval") or {}).get("edge") for x in legs]
        edges = [x for x in edges if x is not None]
        avg_edge = np.mean(edges) if edges else np.nan

        print(f"\n  #{i} {p['type'].upper()}")
        print(f"  {'-' * 106}")
        print(f"  Estimated win probability: {p['probability']:.2%}")
        print(f"  Fair odds:                 {am:+d}")
        print(f"  Combined decimal odds:     {p['dec_odds']:.2f}x")
        print(f"  Estimated EV:              {p['ev']:+.2f}u")
        print(f"  Average leg probability:   {avg_prob:.1%}")
        if not np.isnan(avg_edge):
            print(f"  Average edge:              {avg_edge:+.1%}")

        print()
        print(
            f"  {'Player':<24} {'Pos':<4} {'Team':<5} {'Matchup':<10}"
            f"{'TD Prob':>8} {'Edge':>8} {'Book':<14} {'Odds':>7} {'Form':>8}"
        )
        print("  " + "-" * 100)

        for leg in legs:
            team = str(leg.get("team", ""))[:4]
            opp = str(leg.get("opp", ""))[:4]
            matchup = f"{team}-{opp}" if team and opp else team or "—"
            edge = (leg.get("eval") or {}).get("edge")
            edge_s = "—" if edge is None else f"{edge:+.1%}"

            print(
                f"  {str(leg.get('player',''))[:23]:<24} "
                f"{str(leg.get('pos',''))[:3]:<4} "
                f"{team:<5} {matchup:<10}"
                f"{leg.get('p_td',0):>8.1%} {edge_s:>8} "
                f"{str(leg.get('book','—'))[:14]:<14} "
                f"{_fmt_odds(leg.get('odds')):>7} {_form_label(leg):>8}"
            )

    print("\n  Joint probability still uses the original 0.75 haircut.")
    print("  This is a conservative independence adjustment, not a measured correlation model.")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    import sqlite3
    print(f"{SEP}\n  NFL ANYTIME-TD LIVE — season {CURRENT_SEASON}\n{SEP}")
    b = joblib.load(BUNDLE)
    model, feats = b["model"], b["features"]
    order = list(getattr(model, "feature_names_in_", feats))
    print(f"  model: {b.get('model_type','?')}  |  {len(feats)} features")
    print(f"  cache: {CACHE_DIR}")

    conn = sqlite3.connect(DB)
    hist = build_history(conn)
    conn.close()
    have_current = (hist["season"] == CURRENT_SEASON).any()
    if not have_current:
        print(f"  no {CURRENT_SEASON} games yet — using most recent season "
              f"({FALLBACK_SEASONS[0]}) form for Week 1 projections")

    # ── which week are we predicting? ──
    seas, wk, playing, matchup = resolve_slate(FORCE_WEEK)
    if wk:
        print(f"  upcoming slate: {seas} week {wk}  |  {len(playing)} teams playing")
        if not playing:
            print(f"  (no games returned — odds/slate may not be posted yet)")
    else:
        print(f"  could not resolve the slate from ESPN; scoring all players")

    latest = current_form(hist)
    latest = seed_rookies(latest, feats)

    # Current 2026 ESPN rosters are authoritative for player identity.
    # Historical rows remain untouched as feature/form history.
    latest, current_roster = attach_current_roster(latest, playing, matchup)

    # Remove confirmed OUT/IR/inactive players independently of sportsbook odds.
    latest = filter_unavailable_players(latest)

    X = latest.reindex(columns=order)
    latest["p_td"] = model.predict_proba(X)[:, 1]
    latest["fair_odds"] = latest["p_td"].apply(prob_to_american)
    latest["low_conf"] = (~latest["from_current_season"]) | \
        latest.get("is_rookie_seed", pd.Series(False, index=latest.index)).fillna(False)
    # low_conf is a provenance/display flag. It no longer automatically blocks
    # veteran Week-1 fallback rows from betting; only true rookie seeds are blocked.

    # ── odds + value ──
    print(f"\n  fetching book anytime-TD odds...")
    book = fetch_td_odds()
    print(f"  {len(book)} players priced by the book")

    # book keyed by NORMALIZED name so matching uses the thorough normalizer
    # (suffixes, middle initials, punctuation) — the same one that worked all
    # last season. Rebuild the lookup on normalized keys.
    norm_book = {normalize_player_name(k): v for k, v in book.items()}

    def match_odds(name):
        return norm_book.get(normalize_player_name(name))

    rows = []
    for _, r in latest.iterrows():
        o = match_odds(r["player_name"])
        row = {"player_id": int(r["player_id"]) if pd.notna(r.get("player_id")) else None,
               "player": r["player_name"],
               "team": r.get("current_team", ""),
               "opp": r.get("current_opp", ""),
               "pos": r.get("current_pos", ""),
               "p_td": float(r["p_td"]), "fair": r["fair_odds"],
               "low_conf": bool(r["low_conf"]),
               "is_rookie_seed": (
                   bool(r.get("is_rookie_seed"))
                   if pd.notna(r.get("is_rookie_seed", np.nan))
                   else False
               ),
               "touch8": r.get("roll8_touch", np.nan),
               "ez_car": r.get("r_ez_carries", np.nan),
               "ez_tgt": r.get("r_ez_targets", np.nan),
               "odds": None, "eval": {"should_bet": False}}
        if o:
            row["odds"] = o["price"]; row["book"] = o["book"]
            row["eval"] = evaluate_bet(r["p_td"], o["price"])
        rows.append(row)

    # ── QUALIFIED SINGLES with Kelly sizing ──
    priced = [r for r in rows if r["odds"] is not None]

    # Week 1 handling:
    # - Veterans using 2025 fallback form ARE allowed to qualify.
    # - True rookie projection seeds remain blocked from auto-bet/parlays until
    #   they have real NFL game history.
    bets = [
        r for r in priced
        if r["eval"]["should_bet"]
        and not r.get("is_rookie_seed", False)
    ]
    bets.sort(key=lambda r: -r["eval"]["edge"])
    print_single_bets(bets)

    # ── PARLAYS (3 types, 0.75 haircut) ──
    parlays = generate_parlays(bets)
    print_parlay_table(parlays)

    # Persist the exact pregame decision state for prospective grading.
    pred_records = []
    for r in rows:
        e = r.get("eval") or {}
        rec = {k: v for k, v in r.items() if k != "eval"}
        rec.update({
            "season": int(seas or CURRENT_SEASON),
            "week": int(wk or FORCE_WEEK or 0),
            "line": 0.5,
            "matchup": f"{r.get('team','')}-{r.get('opp','')}",
            "form_source": _form_label(r),
            "implied_prob": e.get("implied"),
            "edge": e.get("edge"),
            "should_bet": bool(e.get("should_bet", False)),
            "stake_units": e.get("stake_units", 0.0),
            "confidence": e.get("confidence", ""),
            "bet_reason": e.get("reason", ""),
        })
        pred_records.append(rec)
    res = pd.DataFrame(pred_records)

    predictions_dir = PROJECT_ROOT / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    week_tag = f"week{int(wk or FORCE_WEEK or 0):02d}"
    snapshot_out = predictions_dir / f"nfl_td_predictions_{int(seas or CURRENT_SEASON)}_{week_tag}.csv"
    latest_out = Path(os.getenv(
        "NFL_TD_PREDICTIONS_OUT",
        str(predictions_dir / f"nfl_td_predictions_{CURRENT_SEASON}.csv"),
    ))
    res.to_csv(snapshot_out, index=False)
    res.to_csv(latest_out, index=False)

    parlay_rows = []
    for pidx, p in enumerate(parlays, 1):
        fair_odds = prob_to_american(p["probability"])
        for lidx, leg in enumerate(p["legs"], 1):
            e = leg.get("eval") or {}
            parlay_rows.append({
                "season": int(seas or CURRENT_SEASON),
                "week": int(wk or FORCE_WEEK or 0),
                "parlay_id": pidx,
                "parlay_type": p.get("type", ""),
                "num_legs": p.get("num_legs", len(p.get("legs", []))),
                "parlay_probability": p.get("probability"),
                "parlay_fair_odds": fair_odds,
                "parlay_dec_odds": p.get("dec_odds"),
                "parlay_ev": p.get("ev"),
                "leg_number": lidx,
                "player_id": leg.get("player_id"),
                "player": leg.get("player"),
                "team": leg.get("team"),
                "opp": leg.get("opp"),
                "pos": leg.get("pos"),
                "line": 0.5,
                "p_td": leg.get("p_td"),
                "odds": leg.get("odds"),
                "book": leg.get("book", ""),
                "edge": e.get("edge"),
                "stake_units": e.get("stake_units", 0.0),
                "confidence": e.get("confidence", ""),
                "form_source": _form_label(leg),
            })
    parlays_out = predictions_dir / f"nfl_td_parlays_{int(seas or CURRENT_SEASON)}_{week_tag}.csv"
    pd.DataFrame(parlay_rows).to_csv(parlays_out, index=False)

    print(f"\n  saved predictions snapshot -> {snapshot_out}")
    print(f"  saved latest predictions   -> {latest_out}")
    print(f"  saved parlays              -> {parlays_out}")

    if not have_current:
        print(f"\n  NOTE: veteran rows use {FALLBACK_SEASONS[0]} fallback form before "
              f"Week 1 but may still qualify if they pass the betting thresholds.")
        print("  True rookie projection seeds remain excluded from auto-bets/parlays "
              "until real NFL history is available.")
    print(SEP)


main()