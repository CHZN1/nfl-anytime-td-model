#!/usr/bin/env python3
"""
Build the SQLite database used by the NFL anytime-TD model.

Sources
-------
ESPN:
    - schedule / game IDs
    - player box scores
    - venue and game context
    - point spread and over/under

nflverse via nfl_data_py:
    - play-by-play TD opportunity features
    - NGS receiving metrics
    - offensive snap share

Season coverage
---------------
ESPN tables:       2020-2025
pbp_td_features:   2021-2025

The 2020 PBP feature season is intentionally omitted to match the database used
for the published model. The trainer therefore begins in 2021.

Notes
-----
This script reproduces the data pipeline used by the current model as closely
as possible from the original collection scripts. Weather fields in the ESPN
collector are venue-based proxies, not observed historical weather.

The PBP/NGS/snap collector uses a player-ID crosswalk containing:
    gsis_id, espn_id
and, when available:
    pfr_id

If the crosswalk file does not exist, this script builds it automatically from
nfl_data_py.import_ids(), checks mapping coverage against player_game_logs, and
saves it for reuse.

Example
-------
python data/build_nfl_database.py \
    --db data/nfl_props_COMPLETE.db \
    --crosswalk data/nfl_id_crosswalk.csv
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

ESPN_SEASONS = list(range(2020, 2026))
PBP_SEASONS = list(range(2021, 2026))

MAX_RETRIES = 3
HOME_RELATIVE_SPREAD = True
CACHE_VERSION = "v2"

log = logging.getLogger("nfl_db")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    retries: int = MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            time.sleep(0.5 + random.uniform(0.0, 0.5))
            response = requests.get(url, params=params, timeout=25)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 404:
                return None

            if response.status_code in {502, 503, 504}:
                wait = 2 ** attempt
            else:
                wait = 2

            log.warning(
                "HTTP %s for %s (attempt %s/%s)",
                response.status_code,
                url,
                attempt + 1,
                retries,
            )
            time.sleep(wait)

        except requests.RequestException as exc:
            log.warning(
                "Request failed for %s (attempt %s/%s): %s",
                url,
                attempt + 1,
                retries,
                exc,
            )
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_game_logs (
            player_id INTEGER,
            player_name TEXT,
            game_id INTEGER,
            season INTEGER,
            week INTEGER,
            team_id INTEGER,
            opponent_id INTEGER,
            home_away TEXT,
            game_date TEXT,
            passing_yards INTEGER,
            passing_tds INTEGER,
            passing_attempts INTEGER,
            completions INTEGER,
            interceptions INTEGER,
            sacks_taken INTEGER,
            passing_rating REAL,
            rushing_yards INTEGER,
            rushing_tds INTEGER,
            rushing_attempts INTEGER,
            longest_rush INTEGER,
            yards_per_carry REAL,
            receiving_yards INTEGER,
            receiving_tds INTEGER,
            receptions INTEGER,
            targets INTEGER,
            longest_reception INTEGER,
            fumbles INTEGER,
            fumbles_lost INTEGER,
            field_goals_made INTEGER,
            field_goals_attempted INTEGER,
            extra_points_made INTEGER,
            target_share REAL,
            usage_rate REAL,
            catch_rate REAL,
            yards_per_target REAL,
            yards_per_reception REAL,
            red_zone_targets INTEGER,
            red_zone_receptions INTEGER,
            red_zone_carries INTEGER,
            goal_line_carries INTEGER,
            third_down_targets INTEGER,
            opportunity_share REAL,
            production_efficiency REAL,
            UNIQUE(player_id, game_id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS game_context (
            game_id INTEGER,
            season INTEGER,
            week INTEGER,
            home_team_id INTEGER,
            away_team_id INTEGER,
            venue TEXT,
            game_date TEXT,
            temperature INTEGER,
            wind_speed INTEGER,
            precipitation REAL,
            dome_game INTEGER,
            weather_impact_score REAL,
            point_spread REAL,
            over_under REAL,
            UNIQUE(game_id)
        )
        """
    )

    conn.commit()


# ---------------------------------------------------------------------------
# ESPN box scores + base game context
# ---------------------------------------------------------------------------

DOME_STADIUMS = {
    "AT&T Stadium",
    "Mercedes-Benz Stadium",
    "Caesars Superdome",
    "Mercedes-Benz Superdome",
    "Ford Field",
    "Lucas Oil Stadium",
    "U.S. Bank Stadium",
    "State Farm Stadium",
    "University of Phoenix Stadium",
    "Allegiant Stadium",
    "SoFi Stadium",
    "NRG Stadium",
}


def weather_proxy(venue: str) -> Dict[str, float]:
    """Venue-based proxy values preserved from the original collector."""
    if venue in DOME_STADIUMS:
        return {
            "temperature": 72,
            "wind_speed": 0,
            "precipitation": 0.0,
            "dome_game": 1,
            "weather_impact_score": 0.0,
        }

    v = venue.lower()

    if any(city in v for city in (
        "green bay", "chicago", "buffalo", "cleveland", "detroit", "minnesota"
    )):
        return {
            "temperature": 45,
            "wind_speed": 12,
            "precipitation": 0.1,
            "dome_game": 0,
            "weather_impact_score": 2.0,
        }

    if any(city in v for city in (
        "miami", "tampa", "arizona", "las vegas", "dallas", "houston"
    )):
        return {
            "temperature": 78,
            "wind_speed": 6,
            "precipitation": 0.0,
            "dome_game": 0,
            "weather_impact_score": 0.5,
        }

    return {
        "temperature": 62,
        "wind_speed": 9,
        "precipitation": 0.0,
        "dome_game": 0,
        "weather_impact_score": 1.0,
    }


def get_schedule(season: int, week: int) -> List[Dict[str, Any]]:
    data = request_json(
        f"{ESPN_SITE}/scoreboard",
        {"dates": season, "seasontype": 2, "week": week},
    )
    if not data:
        return []

    games: List[Dict[str, Any]] = []

    for event in data.get("events", []):
        competitions = event.get("competitions") or [{}]
        comp = competitions[0]
        competitors = comp.get("competitors") or []

        if len(competitors) < 2:
            continue

        home = next(
            (x for x in competitors if x.get("homeAway") == "home"),
            competitors[0],
        )
        away = next(
            (x for x in competitors if x.get("homeAway") == "away"),
            competitors[1],
        )

        try:
            games.append(
                {
                    "game_id": int(event["id"]),
                    "date": event.get("date"),
                    "season": season,
                    "week": week,
                    "home_team": {
                        "id": int(home["team"]["id"]),
                        "name": home["team"].get("displayName", ""),
                    },
                    "away_team": {
                        "id": int(away["team"]["id"]),
                        "name": away["team"].get("displayName", ""),
                    },
                    "venue": comp.get("venue", {}).get("fullName", "Unknown"),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    return games


def safe_int(value: Any, default: int = 0) -> int:
    try:
        s = str(value).strip()
        if s in {"--", "", "N/A", "None"}:
            return default
        if "/" in s or "-" in s:
            return int(float(s.split("/")[0].split("-")[0]))
        return int(float(s))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        s = str(value).strip()
        if s in {"--", "", "N/A", "None"}:
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


def parse_stat_category(player: Dict[str, Any], stat_name: str, stats: List[Any]) -> None:
    if "passing" in stat_name and len(stats) >= 3:
        comp_att = str(stats[0])
        if "/" in comp_att or "-" in comp_att:
            parts = comp_att.replace("/", "-").split("-")
            player["completions"] = safe_int(parts[0])
            player["passing_attempts"] = safe_int(parts[1] if len(parts) > 1 else 0)

        player["passing_yards"] = safe_int(stats[1])
        player["passing_tds"] = safe_int(stats[3] if len(stats) > 3 else 0)
        player["interceptions"] = safe_int(stats[4] if len(stats) > 4 else 0)

    elif "rushing" in stat_name and len(stats) >= 2:
        attempts = safe_int(stats[0])
        yards = safe_int(stats[1])

        player.update(
            {
                "rushing_attempts": attempts,
                "rushing_yards": yards,
                "yards_per_carry": safe_float(stats[2] if len(stats) > 2 else 0),
                "rushing_tds": safe_int(stats[3] if len(stats) > 3 else 0),
                "longest_rush": safe_int(stats[4] if len(stats) > 4 else 0),
            }
        )

        if attempts > 0 and player["yards_per_carry"] == 0:
            player["yards_per_carry"] = round(yards / attempts, 1)

    elif "receiving" in stat_name and len(stats) >= 2:
        receptions = safe_int(stats[0])
        yards = safe_int(stats[1])

        player.update(
            {
                "receptions": receptions,
                "receiving_yards": yards,
                "yards_per_reception": safe_float(stats[2] if len(stats) > 2 else 0),
                "receiving_tds": safe_int(stats[3] if len(stats) > 3 else 0),
                "longest_reception": safe_int(stats[4] if len(stats) > 4 else 0),
                "targets": safe_int(stats[5] if len(stats) > 5 else receptions),
            }
        )

        if receptions > 0 and player["yards_per_reception"] == 0:
            player["yards_per_reception"] = round(yards / receptions, 1)

    elif "fumbles" in stat_name:
        player["fumbles"] = safe_int(stats[0] if stats else 0)
        player["fumbles_lost"] = safe_int(stats[1] if len(stats) > 1 else 0)


def get_player_stats(game_id: int) -> List[Dict[str, Any]]:
    data = request_json(f"{ESPN_SITE}/summary", {"event": game_id})
    if not data:
        return []

    box = data.get("boxscore") or {}
    teams = box.get("players") or []
    if not teams:
        return []

    player_stats: Dict[int, Dict[str, Any]] = {}

    for team_data in teams:
        team_id = team_data.get("team", {}).get("id")
        if not team_id:
            continue

        for category in team_data.get("statistics", []):
            stat_name = str(category.get("name", "")).lower()

            for athlete_data in category.get("athletes", []):
                athlete = athlete_data.get("athlete") or {}
                stats_array = athlete_data.get("stats") or []

                if not athlete.get("id"):
                    continue

                player_id = int(athlete["id"])

                player = player_stats.setdefault(
                    player_id,
                    {
                        "player_id": player_id,
                        "player_name": athlete.get("displayName", f"Player_{player_id}"),
                        "team_id": int(team_id),
                        "game_id": int(game_id),
                        "passing_yards": 0,
                        "passing_tds": 0,
                        "passing_attempts": 0,
                        "completions": 0,
                        "interceptions": 0,
                        "sacks_taken": 0,
                        "passing_rating": 0.0,
                        "rushing_yards": 0,
                        "rushing_tds": 0,
                        "rushing_attempts": 0,
                        "longest_rush": 0,
                        "yards_per_carry": 0.0,
                        "receiving_yards": 0,
                        "receiving_tds": 0,
                        "receptions": 0,
                        "targets": 0,
                        "longest_reception": 0,
                        "fumbles": 0,
                        "fumbles_lost": 0,
                        "yards_per_reception": 0.0,
                    },
                )

                parse_stat_category(player, stat_name, stats_array)

    return list(player_stats.values())


PLAYER_INSERT = """
INSERT OR REPLACE INTO player_game_logs
(
    player_id, player_name, game_id, season, week, team_id, opponent_id,
    home_away, game_date,
    passing_yards, passing_tds, passing_attempts, completions, interceptions,
    sacks_taken, passing_rating,
    rushing_yards, rushing_tds, rushing_attempts, longest_rush, yards_per_carry,
    receiving_yards, receiving_tds, receptions, targets, longest_reception,
    fumbles, fumbles_lost, field_goals_made, field_goals_attempted,
    extra_points_made, target_share, usage_rate, catch_rate, yards_per_target,
    yards_per_reception, red_zone_targets, red_zone_receptions,
    red_zone_carries, goal_line_carries, third_down_targets,
    opportunity_share, production_efficiency
)
VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
"""


def save_game(conn: sqlite3.Connection, game: Dict[str, Any], players: List[Dict[str, Any]]) -> None:
    weather = weather_proxy(game["venue"])

    conn.execute(
        """
        INSERT INTO game_context
        (
            game_id, season, week, home_team_id, away_team_id, venue, game_date,
            temperature, wind_speed, precipitation, dome_game, weather_impact_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            season=excluded.season,
            week=excluded.week,
            home_team_id=excluded.home_team_id,
            away_team_id=excluded.away_team_id,
            venue=excluded.venue,
            game_date=excluded.game_date,
            temperature=excluded.temperature,
            wind_speed=excluded.wind_speed,
            precipitation=excluded.precipitation,
            dome_game=excluded.dome_game,
            weather_impact_score=excluded.weather_impact_score
        """,
        (
            game["game_id"],
            game["season"],
            game["week"],
            game["home_team"]["id"],
            game["away_team"]["id"],
            game["venue"],
            game["date"],
            weather["temperature"],
            weather["wind_speed"],
            weather["precipitation"],
            weather["dome_game"],
            weather["weather_impact_score"],
        ),
    )

    for p in players:
        is_home = p["team_id"] == game["home_team"]["id"]
        opponent_id = game["away_team"]["id"] if is_home else game["home_team"]["id"]
        home_away = "home" if is_home else "away"

        targets = p.get("targets", 0)
        carries = p.get("rushing_attempts", 0)
        receptions = p.get("receptions", 0)
        rec_yards = p.get("receiving_yards", 0)

        usage_rate = ((targets + carries) / 65 * 100) if (targets + carries) > 0 else 0.0
        catch_rate = (receptions / targets * 100) if targets > 0 else 0.0
        yards_per_target = (rec_yards / targets) if targets > 0 else 0.0

        values = (
            p["player_id"], p["player_name"], p["game_id"],
            game["season"], game["week"], p["team_id"], opponent_id,
            home_away, game["date"],
            p.get("passing_yards", 0), p.get("passing_tds", 0),
            p.get("passing_attempts", 0), p.get("completions", 0),
            p.get("interceptions", 0), p.get("sacks_taken", 0),
            p.get("passing_rating", 0.0),
            p.get("rushing_yards", 0), p.get("rushing_tds", 0),
            p.get("rushing_attempts", 0), p.get("longest_rush", 0),
            p.get("yards_per_carry", 0.0),
            p.get("receiving_yards", 0), p.get("receiving_tds", 0),
            p.get("receptions", 0), p.get("targets", 0),
            p.get("longest_reception", 0),
            p.get("fumbles", 0), p.get("fumbles_lost", 0),
            0, 0, 0,
            0.0, usage_rate, catch_rate, yards_per_target,
            p.get("yards_per_reception", 0.0),
            0, 0, 0, 0, 0, 0.0,
            yards_per_target,
        )

        conn.execute(PLAYER_INSERT, values)

    conn.commit()


def collect_espn_games(conn: sqlite3.Connection, seasons: Iterable[int]) -> None:
    log.info("Collecting ESPN player-game and context data")

    for season in seasons:
        season_games = 0
        season_players = 0

        for week in range(1, 19):
            games = get_schedule(season, week)
            if not games:
                break

            completed_this_week = 0

            for game in games:
                players = get_player_stats(game["game_id"])
                if not players:
                    continue

                save_game(conn, game, players)
                season_games += 1
                season_players += len(players)
                completed_this_week += 1

            log.info(
                "%s week %s: %s completed games collected",
                season,
                week,
                completed_this_week,
            )

            if completed_this_week == 0:
                break

        log.info(
            "%s ESPN complete: %s games, %s player rows processed",
            season,
            season_games,
            season_players,
        )


# ---------------------------------------------------------------------------
# ESPN odds backfill
# ---------------------------------------------------------------------------

def home_abbreviation(comp: Dict[str, Any]) -> Optional[str]:
    for c in comp.get("competitors", []):
        if c.get("homeAway") == "home":
            return c.get("team", {}).get("abbreviation")
    return None


def extract_odds_entry(
    entry: Dict[str, Any],
    home_abbr: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    ou = entry.get("overUnder")
    over_under = float(ou) if ou is not None and not isinstance(ou, dict) else None

    spread = entry.get("spread")
    if spread is None or isinstance(spread, dict):
        return None, over_under

    if not HOME_RELATIVE_SPREAD:
        return float(spread), over_under

    magnitude = abs(float(spread))

    home_odds = entry.get("homeTeamOdds") or {}
    away_odds = entry.get("awayTeamOdds") or {}

    home_favorite: Optional[bool] = None

    if home_odds.get("favorite") is True:
        home_favorite = True
    elif away_odds.get("favorite") is True:
        home_favorite = False

    if home_favorite is None:
        details = str(entry.get("details", "")).strip()
        favorite_abbr = details.split()[0] if details and details[0].isalpha() else None
        if favorite_abbr and home_abbr:
            home_favorite = favorite_abbr == home_abbr

    if home_favorite is None:
        return None, over_under

    home_spread = -magnitude if home_favorite else magnitude
    return home_spread, over_under


def parse_site_odds(
    summary: Dict[str, Any],
    home_abbr: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    for key in ("pickcenter", "odds"):
        entries = summary.get(key) or []
        if not entries:
            continue

        entry = next(
            (
                e for e in entries
                if str(e.get("provider", {}).get("name", "")).lower() == "consensus"
            ),
            entries[0],
        )

        spread, total = extract_odds_entry(entry, home_abbr)
        if spread is not None or total is not None:
            return spread, total

    return None, None


def fetch_core_odds(
    game_id: int,
    home_abbr: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    data = request_json(
        f"{ESPN_CORE}/events/{game_id}/competitions/{game_id}/odds"
    )
    items = (data or {}).get("items") or []
    if not items:
        return None, None

    ordered = sorted(
        items,
        key=lambda o: o.get("provider", {}).get("priority", 99),
    )

    spread: Optional[float] = None
    total: Optional[float] = None

    for entry in ordered:
        if spread is None:
            s = entry.get("spread")
            if s is not None and not isinstance(s, dict):
                parsed_spread, _ = extract_odds_entry(entry, home_abbr)
                if parsed_spread is not None:
                    spread = parsed_spread

        if total is None:
            ou = entry.get("overUnder")
            if ou is not None and not isinstance(ou, dict):
                total = float(ou)

        if spread is not None and total is not None:
            break

    return spread, total


def collect_espn_odds(conn: sqlite3.Connection, seasons: Iterable[int]) -> None:
    log.info("Backfilling ESPN spreads and totals")

    for season in seasons:
        updated = 0

        for week in range(1, 19):
            data = request_json(
                f"{ESPN_SITE}/scoreboard",
                {"dates": season, "seasontype": 2, "week": week},
            )
            events = (data or {}).get("events") or []
            if not events:
                break

            for event in events:
                try:
                    game_id = int(event["id"])
                except (KeyError, TypeError, ValueError):
                    continue

                row = conn.execute(
                    """
                    SELECT point_spread, over_under
                    FROM game_context
                    WHERE game_id=? AND season=?
                    """,
                    (game_id, season),
                ).fetchone()

                if row is None:
                    continue

                current_spread, current_total = row

                if (
                    current_spread not in (None, 0.0)
                    and current_total not in (None, 0.0)
                ):
                    continue

                comp = (event.get("competitions") or [{}])[0]
                home_abbr = home_abbreviation(comp)

                summary = request_json(f"{ESPN_SITE}/summary", {"event": game_id}) or {}
                spread, total = parse_site_odds(summary, home_abbr)

                if spread is None or total is None:
                    core_spread, core_total = fetch_core_odds(game_id, home_abbr)
                    if spread is None:
                        spread = core_spread
                    if total is None:
                        total = core_total

                assignments = []
                values: List[Any] = []

                if spread is not None:
                    assignments.append("point_spread=?")
                    values.append(float(spread))

                if total is not None:
                    assignments.append("over_under=?")
                    values.append(float(total))

                if not assignments:
                    continue

                values.extend([game_id, season])

                conn.execute(
                    f"""
                    UPDATE game_context
                    SET {", ".join(assignments)}
                    WHERE game_id=? AND season=?
                    """,
                    values,
                )
                updated += 1

            conn.commit()

        log.info("%s odds complete: %s game rows updated", season, updated)


# ---------------------------------------------------------------------------
# nfl_data_py PBP + NGS + snaps
# ---------------------------------------------------------------------------

def build_or_load_crosswalk(
    path: Path,
    db_path: Path,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load the ESPN/GSIS/PFR crosswalk, or build it from nfl_data_py.import_ids()."""
    try:
        import nfl_data_py as nfl
    except ImportError as exc:
        raise RuntimeError(
            "nfl_data_py is required to build or use the PBP feature pipeline."
        ) from exc

    if path.exists():
        xwalk = pd.read_csv(path)
        log.info("Loaded player-ID crosswalk: %s", path)
    else:
        log.info("Crosswalk not found; building it from nfl_data_py.import_ids()")
        ids = nfl.import_ids()

        required = {"espn_id", "gsis_id"}
        missing = required.difference(ids.columns)
        if missing:
            raise ValueError(
                "nfl_data_py.import_ids() is missing required columns: "
                f"{sorted(missing)}"
            )

        keep = ["espn_id", "gsis_id"]
        if "pfr_id" in ids.columns:
            keep.append("pfr_id")

        xwalk = ids[keep].dropna(subset=["espn_id"]).copy()
        xwalk["espn_id"] = pd.to_numeric(xwalk["espn_id"], errors="coerce")
        xwalk = xwalk.dropna(subset=["espn_id", "gsis_id"]).copy()
        xwalk["espn_id"] = xwalk["espn_id"].astype("int64")

        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }

                if "player_game_logs" in tables:
                    db_ids = pd.read_sql_query(
                        "SELECT DISTINCT player_id FROM player_game_logs",
                        conn,
                    )
                    db_ids["player_id"] = pd.to_numeric(
                        db_ids["player_id"],
                        errors="coerce",
                    )

                    valid = db_ids["player_id"].dropna()
                    espn_ids = set(xwalk["espn_id"])

                    if len(valid):
                        direct = valid.isin(espn_ids).mean()
                        abs_match = valid.abs().isin(espn_ids).mean()
                        best = "abs(player_id)" if abs_match > direct else "player_id"
                        best_cov = max(direct, abs_match)

                        log.info(
                            "Crosswalk coverage: direct %.1f%% | abs(id) %.1f%% | best %s %.1f%%",
                            direct * 100,
                            abs_match * 100,
                            best,
                            best_cov * 100,
                        )

                        if abs_match > direct + 0.05:
                            log.warning(
                                "abs(player_id) maps materially better than direct IDs. "
                                "Inspect the source database before building PBP features."
                            )

        path.parent.mkdir(parents=True, exist_ok=True)
        xwalk.to_csv(path, index=False)
        log.info("Saved player-ID crosswalk: %s", path)

    required = {"gsis_id", "espn_id"}
    missing = required.difference(xwalk.columns)
    if missing:
        raise ValueError(
            f"Crosswalk is missing required columns: {sorted(missing)}"
        )

    xwalk["espn_id"] = pd.to_numeric(xwalk["espn_id"], errors="coerce")
    xwalk = xwalk.dropna(subset=["espn_id", "gsis_id"]).copy()
    xwalk["espn_id"] = xwalk["espn_id"].astype("int64")

    gsis_to_espn = dict(zip(xwalk["gsis_id"], xwalk["espn_id"]))

    pfr_to_espn: Dict[str, int] = {}
    if "pfr_id" in xwalk.columns:
        pfr = xwalk.dropna(subset=["pfr_id"])
        pfr_to_espn = dict(zip(pfr["pfr_id"], pfr["espn_id"]))

    return gsis_to_espn, pfr_to_espn

def numeric_column(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def build_pbp_features(
    db_path: Path,
    crosswalk_path: Path,
    cache_dir: Path,
    seasons: Iterable[int],
) -> None:
    try:
        import nfl_data_py as nfl
    except ImportError as exc:
        raise RuntimeError(
            "nfl_data_py is required for pbp_td_features. "
            "Install it in the environment used for data collection."
        ) from exc

    gsis_to_espn, pfr_to_espn = build_or_load_crosswalk(crosswalk_path, db_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    all_player_games: List[pd.DataFrame] = []

    log.info(
        "Building PBP/NGS/snap features (%s GSIS mappings, %s PFR mappings)",
        len(gsis_to_espn),
        len(pfr_to_espn),
    )

    for season in seasons:
        cache_path = cache_dir / f"pbp_td_{CACHE_VERSION}_{season}.parquet"

        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
                ids_ok = (
                    pd.to_numeric(cached["player_id"], errors="coerce")
                    .notna()
                    .mean()
                    > 0.5
                )
                if ids_ok:
                    all_player_games.append(cached)
                    log.info("%s PBP: loaded %s cached rows", season, len(cached))
                    continue
            except Exception:
                pass

            cache_path.unlink(missing_ok=True)

        pbp = nfl.import_pbp_data([season], downcast=True, cache=False)
        if pbp is None or pbp.empty:
            log.warning("%s PBP: no data", season)
            continue

        rec = (
            pbp[pbp["receiver_player_id"].notna()].copy()
            if "receiver_player_id" in pbp.columns
            else pbp.iloc[0:0].copy()
        )

        if not rec.empty:
            yardline = pd.to_numeric(rec["yardline_100"], errors="coerce")
            rec["_ez"] = (yardline <= 20).astype(float)
            rec["_i10"] = (yardline <= 10).astype(float)
            rec["_i5"] = (yardline <= 5).astype(float)
            rec["_ay"] = pd.to_numeric(rec.get("air_yards"), errors="coerce")
            rec["_tdp"] = pd.to_numeric(rec.get("td_prob"), errors="coerce")

            recg = (
                rec.groupby(["receiver_player_id", "game_id", "posteam"])
                .agg(
                    targets=("receiver_player_id", "size"),
                    ez_targets=("_ez", "sum"),
                    inside10_targets=("_i10", "sum"),
                    inside5_targets=("_i5", "sum"),
                    air_yards_game=("_ay", "sum"),
                    sum_td_prob=("_tdp", "sum"),
                )
                .reset_index()
                .rename(
                    columns={
                        "receiver_player_id": "player_id",
                        "posteam": "team",
                    }
                )
            )
            recg["role"] = "rec"
            recg["adot_game"] = (
                recg["air_yards_game"]
                / recg["targets"].replace(0, np.nan)
            )
        else:
            recg = pd.DataFrame()

        rush = (
            pbp[pbp["rusher_player_id"].notna()].copy()
            if "rusher_player_id" in pbp.columns
            else pbp.iloc[0:0].copy()
        )

        if not rush.empty:
            yardline = pd.to_numeric(rush["yardline_100"], errors="coerce")
            goal_to_go = pd.to_numeric(rush.get("goal_to_go"), errors="coerce")

            rush["_ez"] = ((yardline <= 10) & (goal_to_go == 1)).astype(float)
            rush["_i5"] = (yardline <= 5).astype(float)
            rush["_tdp"] = pd.to_numeric(rush.get("td_prob"), errors="coerce")

            rushg = (
                rush.groupby(["rusher_player_id", "game_id", "posteam"])
                .agg(
                    carries=("rusher_player_id", "size"),
                    ez_carries=("_ez", "sum"),
                    inside5_carries=("_i5", "sum"),
                    rush_sum_td_prob=("_tdp", "sum"),
                )
                .reset_index()
                .rename(
                    columns={
                        "rusher_player_id": "player_id",
                        "posteam": "team",
                    }
                )
            )
            rushg["role"] = "rush"
        else:
            rushg = pd.DataFrame()

        pg = pd.concat([recg, rushg], ignore_index=True)
        pg["season"] = season

        if not recg.empty:
            team_ez = (
                recg.groupby(["team", "game_id"])["ez_targets"]
                .sum()
                .rename("team_ez_targets")
                .reset_index()
            )
            pg = pg.merge(team_ez, on=["team", "game_id"], how="left")
            pg["ez_target_share"] = (
                pg["ez_targets"]
                / pg["team_ez_targets"].replace(0, np.nan)
            )

        try:
            ngs = nfl.import_ngs_data("receiving", [season])
            ngs = ngs[ngs["week"] > 0]

            sched = nfl.import_schedules([season])[
                ["game_id", "week", "home_team", "away_team"]
            ]

            ngs_key = ngs.rename(
                columns={
                    "player_gsis_id": "player_id",
                    "team_abbr": "team",
                }
            )[
                [
                    "player_id",
                    "team",
                    "week",
                    "avg_separation",
                    "avg_cushion",
                    "avg_intended_air_yards",
                    "percent_share_of_intended_air_yards",
                ]
            ]

            long_sched = pd.concat(
                [
                    sched.rename(columns={"home_team": "team"})[
                        ["game_id", "week", "team"]
                    ],
                    sched.rename(columns={"away_team": "team"})[
                        ["game_id", "week", "team"]
                    ],
                ]
            )

            ngs_key = ngs_key.merge(
                long_sched,
                on=["team", "week"],
                how="left",
            ).rename(
                columns={
                    "avg_separation": "ngs_separation",
                    "avg_cushion": "ngs_cushion",
                    "avg_intended_air_yards": "ngs_intended_air",
                    "percent_share_of_intended_air_yards": "ngs_pct_air_share",
                }
            )

            pg = pg.merge(
                ngs_key.drop(columns=["team", "week"]),
                on=["player_id", "game_id"],
                how="left",
            )

        except Exception as exc:
            log.warning("%s NGS skipped: %s", season, exc)

        try:
            snaps = nfl.import_snap_counts([season])
            pfr_col = next(
                (c for c in ("pfr_player_id", "pfr_id") if c in snaps.columns),
                None,
            )

            if pfr_col and "game_id" in snaps.columns and pfr_to_espn:
                snap_key = snaps[[pfr_col, "game_id", "offense_pct"]].copy()
                snap_key["player_id"] = snap_key[pfr_col].map(pfr_to_espn)
                snap_key = snap_key.dropna(subset=["player_id"])
                snap_key["player_id"] = snap_key["player_id"].astype("int64")
                snap_key["game_id"] = snap_key["game_id"].astype(str)
                pg["game_id"] = pg["game_id"].astype(str)

                pg = pg.merge(
                    snap_key[
                        ["player_id", "game_id", "offense_pct"]
                    ].drop_duplicates(["player_id", "game_id"]),
                    on=["player_id", "game_id"],
                    how="left",
                )

        except Exception as exc:
            log.warning("%s snap share skipped: %s", season, exc)

        pg["player_id"] = pg["player_id"].map(gsis_to_espn)
        before = len(pg)
        pg = pg[pg["player_id"].notna()].copy()
        pg["player_id"] = pg["player_id"].astype("int64")

        pg.to_parquet(cache_path, index=False)
        all_player_games.append(pg)

        log.info(
            "%s PBP complete: %s/%s rows mapped to ESPN IDs",
            season,
            len(pg),
            before,
        )

    if not all_player_games:
        raise RuntimeError("No pbp_td_features rows were built.")

    output = pd.concat(all_player_games, ignore_index=True)

    with sqlite3.connect(db_path) as conn:
        output.to_sql(
            "pbp_td_features",
            conn,
            if_exists="replace",
            index=False,
        )

    log.info(
        "pbp_td_features complete: %s rows across %s seasons",
        len(output),
        output["season"].nunique(),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def print_database_summary(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """,
            conn,
        )

        log.info("Database summary: %s", db_path)

        for table in tables["name"]:
            count = conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            log.info("  %-28s %8s rows", table, f"{count:,}")

        for table in ("player_game_logs", "game_context", "pbp_td_features"):
            if table not in set(tables["name"]):
                continue

            cols = {
                row[1]
                for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            }
            if "season" not in cols:
                continue

            rows = conn.execute(
                f"""
                SELECT season, COUNT(*)
                FROM "{table}"
                GROUP BY season
                ORDER BY season
                """
            ).fetchall()

            season_text = ", ".join(
                f"{season}:{count:,}" for season, count in rows
            )
            log.info("  %s seasons -> %s", table, season_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SQLite data used by the NFL anytime-TD model."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "data" / "nfl_props_COMPLETE.db",
        help="SQLite output path.",
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=PROJECT_ROOT / "data" / "nfl_id_crosswalk.csv",
        help="GSIS/PFR -> ESPN player-ID crosswalk. Built automatically if missing.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "pbp",
        help="Parquet cache directory for nfl_data_py PBP aggregation.",
    )
    parser.add_argument(
        "--skip-espn",
        action="store_true",
        help="Do not collect ESPN schedule, box-score, or context data.",
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Do not backfill ESPN point spreads and game totals.",
    )
    parser.add_argument(
        "--skip-pbp",
        action="store_true",
        help="Do not build nfl_data_py PBP/NGS/snap features.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the current database summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.db = args.db.expanduser().resolve()
    args.crosswalk = args.crosswalk.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()

    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        if not args.db.exists():
            raise FileNotFoundError(args.db)
        print_database_summary(args.db)
        return

    with sqlite3.connect(args.db) as conn:
        ensure_schema(conn)

        if not args.skip_espn:
            collect_espn_games(conn, ESPN_SEASONS)

        if not args.skip_odds:
            collect_espn_odds(conn, ESPN_SEASONS)

    if not args.skip_pbp:
        build_pbp_features(
            args.db,
            args.crosswalk,
            args.cache_dir,
            PBP_SEASONS,
        )

    print_database_summary(args.db)


if __name__ == "__main__":
    main()
