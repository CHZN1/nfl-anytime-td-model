#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.getenv(
    "NFL_TD_DB",
    str(PROJECT_ROOT / "data" / "nfl_props_COMPLETE.db"),
))

SEP = "=" * 86

REQUIRED_TABLES = {
    "player_game_logs": {
        "player_id",
        "player_name",
        "game_id",
        "season",
        "week",
        "team_id",
        "opponent_id",
        "rushing_tds",
        "receiving_tds",
        "rushing_attempts",
        "targets",
    },
    "game_context": {
        "game_id",
        "season",
        "week",
        "temperature",
        "dome_game",
        "weather_impact_score",
        "point_spread",
        "over_under",
    },
    "pbp_td_features": {
        "player_id",
        "game_id",
        "season",
        "targets",
        "ez_targets",
        "inside10_targets",
        "inside5_targets",
        "air_yards_game",
        "adot_game",
        "sum_td_prob",
        "ez_target_share",
        "carries",
        "ez_carries",
        "inside5_carries",
        "rush_sum_td_prob",
        "ngs_separation",
        "ngs_cushion",
        "ngs_intended_air",
        "ngs_pct_air_share",
        "offense_pct",
    },
}

OPTIONAL_TABLES = {
    "defense_qb_pressure_stats",
    "defense_vs_position_stats",
    "receiving_advanced_stats",
    "receiving_red_zone_stats",
}


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {r[1] for r in rows}


def row_count(conn: sqlite3.Connection, table: str) -> int:
    return int(
        conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    )


def season_counts(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    if "season" not in table_columns(conn, table):
        return pd.DataFrame()

    return pd.read_sql_query(
        f'''
        SELECT season, COUNT(*) AS rows
        FROM "{table}"
        GROUP BY season
        ORDER BY season
        ''',
        conn,
    )


def duplicate_groups(
    conn: sqlite3.Connection,
    table: str,
    keys: list[str],
) -> int | None:
    cols = table_columns(conn, table)
    if not set(keys).issubset(cols):
        return None

    key_sql = ", ".join(f'"{k}"' for k in keys)
    query = f'''
        SELECT COUNT(*)
        FROM (
            SELECT {key_sql}, COUNT(*) AS n
            FROM "{table}"
            GROUP BY {key_sql}
            HAVING COUNT(*) > 1
        )
    '''
    return int(conn.execute(query).fetchone()[0])


def validate_database(db_path: Path) -> bool:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"{SEP}\n  NFL TD DATABASE VALIDATION\n{SEP}")
    print(f"  database: {db_path}")
    print(f"  size:     {db_path.stat().st_size / (1024 ** 2):.2f} MB\n")

    conn = sqlite3.connect(db_path)

    try:
        existing = table_names(conn)
        failed = False

        print("Required tables")
        print("-" * 60)

        for table, required_cols in REQUIRED_TABLES.items():
            if table not in existing:
                print(f"  FAIL  {table}: missing table")
                failed = True
                continue

            cols = table_columns(conn, table)
            missing = sorted(required_cols - cols)
            count = row_count(conn, table)

            if missing:
                print(
                    f"  FAIL  {table}: {count:,} rows | "
                    f"missing columns: {', '.join(missing)}"
                )
                failed = True
            else:
                print(
                    f"  PASS  {table}: {count:,} rows | {len(cols)} columns"
                )

        print("\nSeason coverage")
        print("-" * 60)

        for table in REQUIRED_TABLES:
            if table not in existing:
                continue

            counts = season_counts(conn, table)
            if counts.empty:
                print(f"  {table}: no season column")
                continue

            summary = ", ".join(
                f"{int(r.season)}={int(r.rows):,}"
                for r in counts.itertuples(index=False)
                if pd.notna(r.season)
            )
            print(f"  {table}: {summary}")

        print("\nKey integrity checks")
        print("-" * 60)

        checks = [
            ("player_game_logs", ["player_id", "game_id"]),
            ("game_context", ["game_id"]),
            ("pbp_td_features", ["player_id", "game_id", "role"]),
        ]

        for table, keys in checks:
            if table not in existing:
                continue

            ndup = duplicate_groups(conn, table, keys)

            if ndup is None and table == "pbp_td_features":
                keys = ["player_id", "game_id"]
                ndup = duplicate_groups(conn, table, keys)

            if ndup is None:
                print(
                    f"  WARN  {table}: cannot test duplicate key "
                    f"({', '.join(keys)})"
                )
            elif ndup == 0:
                print(
                    f"  PASS  {table}: no duplicate "
                    f"({', '.join(keys)}) groups"
                )
            else:
                print(
                    f"  WARN  {table}: {ndup:,} duplicate "
                    f"({', '.join(keys)}) groups"
                )

        print("\nJoin sanity checks")
        print("-" * 60)

        if {"player_game_logs", "game_context"}.issubset(existing):
            query = '''
                SELECT
                    COUNT(*) AS player_rows,
                    SUM(CASE WHEN g.game_id IS NULL THEN 1 ELSE 0 END)
                        AS missing_context
                FROM player_game_logs p
                LEFT JOIN game_context g
                    ON p.game_id = g.game_id
            '''
            player_rows, missing_context = conn.execute(query).fetchone()
            player_rows = int(player_rows or 0)
            missing_context = int(missing_context or 0)
            pct = missing_context / player_rows if player_rows else 0.0
            status = "PASS" if missing_context == 0 else "WARN"

            print(
                f"  {status}  player_game_logs -> game_context: "
                f"{missing_context:,}/{player_rows:,} missing ({pct:.2%})"
            )

        if "pbp_td_features" in existing:
            pbp_cols = table_columns(conn, "pbp_td_features")
            if "week" in pbp_cols:
                print("  PASS  pbp_td_features contains season/week join keys")
            else:
                print(
                    "  INFO  pbp_td_features week is derived from game_id "
                    "inside the trainer/live pipeline"
                )

        print("\nOptional source tables")
        print("-" * 60)

        for table in sorted(OPTIONAL_TABLES):
            if table in existing:
                print(f"  present  {table}: {row_count(conn, table):,} rows")
            else:
                print(f"  absent   {table}")

        print(f"\n{SEP}")
        if failed:
            print("  VALIDATION FAILED — required schema is incomplete")
            print(SEP)
            return False

        print("  VALIDATION PASSED — required trainer/live schema is present")
        print(SEP)
        return True

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the NFL anytime-TD SQLite database."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to nfl_props_COMPLETE.db",
    )
    args = parser.parse_args()

    ok = validate_database(args.db)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
