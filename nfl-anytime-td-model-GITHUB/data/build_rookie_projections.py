#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nfl_data_py as nfl
except ImportError as exc:
    raise SystemExit(
        "nfl_data_py is required. Install dependencies from requirements.txt."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CROSSWALK = Path(os.getenv(
    "NFL_TD_CROSSWALK",
    str(PROJECT_ROOT / "data" / "nfl_id_crosswalk.csv"),
))
DEFAULT_OUT = Path(os.getenv(
    "NFL_TD_ROOKIE_PROJECTIONS",
    str(PROJECT_ROOT / "data" / "rookie_touch_projections.csv"),
))

SEP = "=" * 82

BANDS = {
    ("RB", 1): dict(touch=18.0, ez_car=2.2, ez_tgt=0.4),
    ("RB", 2): dict(touch=9.0, ez_car=0.8, ez_tgt=0.3),
    ("RB", 3): dict(touch=4.0, ez_car=0.3, ez_tgt=0.2),
    ("WR", 1): dict(touch=8.0, ez_car=0.0, ez_tgt=1.4),
    ("WR", 2): dict(touch=6.0, ez_car=0.0, ez_tgt=0.8),
    ("WR", 3): dict(touch=3.0, ez_car=0.0, ez_tgt=0.3),
    ("TE", 1): dict(touch=5.0, ez_car=0.0, ez_tgt=1.0),
    ("TE", 2): dict(touch=2.0, ez_car=0.0, ez_tgt=0.4),
    ("TE", 3): dict(touch=1.5, ez_car=0.0, ez_tgt=0.2),
}

DRAFT_MULTIPLIER = {
    1: 1.20,
    2: 1.10,
    3: 1.00,
    4: 0.90,
    5: 0.85,
}


def usage_band(position: str, depth_rank, draft_round) -> dict[str, float]:
    pos = str(position).upper().strip()

    try:
        rank = int(depth_rank)
    except (TypeError, ValueError):
        rank = 3
    rank = min(max(rank, 1), 3)

    key = (pos, rank)
    if key not in BANDS:
        key = (pos, 3) if pos in {"RB", "WR", "TE"} else ("WR", 3)

    base = dict(BANDS.get(key, dict(touch=2.0, ez_car=0.0, ez_tgt=0.2)))

    try:
        rnd = int(draft_round)
    except (TypeError, ValueError):
        rnd = 6

    mult = DRAFT_MULTIPLIER.get(rnd, 0.75)
    return {k: round(v * mult, 2) for k, v in base.items()}


def build_rookie_projections(season: int, crosswalk_path: Path) -> pd.DataFrame:
    if not crosswalk_path.exists():
        raise FileNotFoundError(
            f"Crosswalk not found: {crosswalk_path}\n"
            "Build data/nfl_id_crosswalk.csv first."
        )

    draft = nfl.import_draft_picks([season])
    required = {"position", "round", "gsis_id"}
    missing = required - set(draft.columns)
    if missing:
        raise ValueError(f"draft_picks missing required columns: {sorted(missing)}")

    draft = draft[draft["position"].isin(["RB", "WR", "TE"])].copy()
    draft = draft.dropna(subset=["gsis_id"])

    name_col = next(
        (c for c in ("pfr_player_name", "player_name", "name") if c in draft.columns),
        None,
    )
    if name_col is None:
        raise ValueError("draft_picks has no recognized player-name column")

    print(f"  {len(draft)} rookie RB/WR/TE from draft_picks")

    depth = nfl.import_depth_charts([season])
    rank = None

    if len(depth) and {"gsis_id", "pos_rank"}.issubset(depth.columns):
        latest_dt = depth["dt"].max() if "dt" in depth.columns else None
        current = depth[depth["dt"] == latest_dt] if latest_dt is not None else depth
        rank = (
            current.dropna(subset=["gsis_id"])
            .groupby("gsis_id")["pos_rank"]
            .min()
            .rename("depth_rank")
        )
        suffix = f" (as of {latest_dt})" if latest_dt is not None else ""
        print(f"  depth chart: {len(rank)} players ranked{suffix}")
    else:
        print("  depth chart ranks unavailable — using draft-round fallback")

    if rank is not None:
        draft = draft.merge(rank, on="gsis_id", how="left")
    else:
        draft["depth_rank"] = np.nan

    draft["depth_rank"] = draft["depth_rank"].where(
        draft["depth_rank"].notna(),
        np.where(pd.to_numeric(draft["round"], errors="coerce") <= 1, 1, 2),
    )

    projected = draft.apply(
        lambda r: pd.Series(usage_band(r["position"], r["depth_rank"], r["round"])),
        axis=1,
    )

    if "team" not in draft.columns:
        draft["team"] = ""

    out = pd.concat(
        [
            draft[[name_col, "position", "team", "round", "depth_rank", "gsis_id"]]
            .reset_index(drop=True),
            projected.reset_index(drop=True),
        ],
        axis=1,
    )

    out = out.rename(columns={
        name_col: "player_name",
        "touch": "proj_touches",
        "ez_car": "proj_ez",
        "ez_tgt": "proj_ez_tgt",
    })

    xw = pd.read_csv(crosswalk_path)
    required_xw = {"gsis_id", "espn_id"}
    missing_xw = required_xw - set(xw.columns)
    if missing_xw:
        raise ValueError(f"crosswalk missing required columns: {sorted(missing_xw)}")

    xw["espn_id"] = pd.to_numeric(xw["espn_id"], errors="coerce")
    g2e = dict(zip(xw["gsis_id"], xw["espn_id"]))
    out["player_id"] = out["gsis_id"].map(g2e)

    mapped = out["player_id"].notna().mean() if len(out) else 0.0
    print(f"  gsis -> ESPN mapped: {mapped:.0%}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build conservative rookie usage priors for the live TD model."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"{SEP}\n  ROOKIE PROJECTIONS — {args.season}\n{SEP}")

    projections = build_rookie_projections(args.season, args.crosswalk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    projections.to_csv(args.out, index=False)

    print(f"\n  wrote {len(projections)} rookie projections -> {args.out}\n")
    print(
        f"  {'player':<24}{'pos':>4}{'tm':>5}{'rd':>4}{'depth':>7}"
        f"{'touch':>7}{'ezCar':>7}{'ezTgt':>7}"
    )

    preview = projections.sort_values("proj_touches", ascending=False).head(15)
    for _, r in preview.iterrows():
        print(
            f"  {str(r['player_name'])[:23]:<24}"
            f"{str(r['position']):>4}"
            f"{str(r.get('team', '')):>5}"
            f"{int(r['round']):>4}"
            f"{float(r['depth_rank']):>7.0f}"
            f"{float(r['proj_touches']):>7.1f}"
            f"{float(r['proj_ez']):>7.2f}"
            f"{float(r['proj_ez_tgt']):>7.2f}"
        )

    print("\n  These seed rookie usage before real NFL history exists.")
    print(SEP)


if __name__ == "__main__":
    main()
