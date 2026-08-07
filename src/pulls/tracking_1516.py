"""Phase 1.1a — extract per-shot difficulty features from the 2015-16 raw SportVU archive.

Why this matters
----------------
Per-shot difficulty features (defender distance, shot clock, touch time, dribbles) are
public for exactly ONE season via the Kaggle 2014-15 shot logs. But the NBA briefly
published *raw* optical tracking for 2015-16 before locking public access down, and the
community preserved it (github.com/sealneaward/nba-movement-data, MIT). Raw tracking is
richer than the Kaggle set: we can *derive* those features ourselves, plus features
neither official release exposes.

Deriving them gives SMOKE a genuine SECOND tracking-rich season -> a real year-over-year
stability test instead of a single-season snapshot.

Data shape (verified 2026-07-20)
--------------------------------
One .7z per game (~6 MB) -> one JSON (~104 MB). 636 games => ~66 GB uncompressed, so we
process game-by-game and keep only the compact per-shot output, never the raw frames.

    game: {gameid, gamedate, events: [...]}
    event: {eventId, visitor, home, moments: [...]}     # eventId joins to PBP EVENTNUM
    moment: [quarter, ts_ms, game_clock, shot_clock, None, entities]
    entity: [team_id, player_id, x, y, z]               # ball is team_id/player_id == -1

Run a single-game demo:
    .venv/Scripts/python.exe -m src.pulls.tracking_1516
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import pandas as pd
import requests

from src.pulls._net import bootstrap, call

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = os.path.join("data", "v2", "raw", "tracking_1516")
OUT_DIR = os.path.join("data", "v2", "processed")
GH_RAW = "https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data"
GH_API = "https://api.github.com/repos/sealneaward/nba-movement-data/contents/data"

# NBA court is 94 x 50 ft in tracking coordinates; hoop centers:
HOOPS = ((5.25, 25.0), (88.75, 25.0))

BALL_ID = -1
# release detection (calibrated against NBA reported shot distance — see find_release)
HELD_DIST_FT = 2.5        # ball this close to the shooter == still in their hands
RELEASE_WINDOW_S = 5.0    # search window around the PBP clock


# --------------------------------------------------------------------------- io
GAME_LIST_CACHE = os.path.join(RAW_DIR, "_game_list.txt")


def list_archive_games() -> list[str]:
    """All .7z filenames in the archive.

    Uses the git trees API (ONE request for the whole repo — the contents API paginates
    and burns through GitHub's 60/hr unauthenticated rate limit), and caches the list on
    disk so repeat runs make zero API calls.
    """
    if os.path.exists(GAME_LIST_CACHE):
        with open(GAME_LIST_CACHE) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names

    bootstrap()
    r = requests.get(
        "https://api.github.com/repos/sealneaward/nba-movement-data/git/trees/master",
        params={"recursive": "1"},
        timeout=60,
    )
    if r.status_code == 403:
        # unauthenticated GitHub API limit (60/hr). Fall back to whatever is already
        # downloaded locally so validation work can proceed; the full list can be
        # cached later once the limit resets.
        local = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".7z")) if os.path.isdir(RAW_DIR) else []
        if local:
            print(f"  (GitHub API rate-limited; using {len(local)} locally cached game(s))")
            return local
        r.raise_for_status()
    r.raise_for_status()
    tree = r.json().get("tree", [])
    names = sorted(
        os.path.basename(t["path"])
        for t in tree
        if t["path"].startswith("data/") and t["path"].endswith(".7z")
    )
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(GAME_LIST_CACHE, "w") as f:
        f.write("\n".join(names))
    return names


_7Z_MAGIC = b"7z\xbc\xaf\x27\x1c"


def _has_valid_7z_header(path: str) -> bool:
    """Just the magic-byte check — NOT a size check.

    Correction (2026-07-20): an earlier version of this function also rejected files
    under 100KB as "truncated downloads." That was wrong — 4/636 games in the source
    repo are genuinely tiny (32-2350 byte) 7z archives that are valid but EMPTY (no
    tracking data was ever recorded for that game upstream). Repeated re-downloads
    returned byte-identical content with HTTP 200, proving these are real files, not
    network truncation. Only the magic bytes actually indicate corruption.
    """
    if not os.path.exists(path) or os.path.getsize(path) < len(_7Z_MAGIC):
        return False
    with open(path, "rb") as f:
        return f.read(len(_7Z_MAGIC)) == _7Z_MAGIC


def download_game(filename: str) -> str:
    """Download one game's .7z (cached, header-validated). Returns local path."""
    bootstrap()
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path) and not _has_valid_7z_header(path):
        os.remove(path)  # drop a genuinely corrupt cache entry and re-fetch
    if not os.path.exists(path):
        r = requests.get(f"{GH_RAW}/{filename}", timeout=300)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        if not _has_valid_7z_header(path):
            n_bytes = os.path.getsize(path)
            os.remove(path)
            raise OSError(f"download of {filename} failed validation ({n_bytes} bytes, bad header)")
    return path


class EmptyArchiveError(Exception):
    """The .7z is well-formed but contains zero files — a genuine upstream data gap."""


def load_game(path: str) -> dict:
    """Extract the .7z to a temp dir and parse the JSON (raw is discarded on exit).

    Raises EmptyArchiveError (not IndexError) if the archive is valid but holds no
    files — a real, if rare, state in this source repo (see download_game docstring).
    """
    import py7zr

    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(path) as z:
            names = z.getnames()
            if not names:
                raise EmptyArchiveError(f"{os.path.basename(path)}: archive contains 0 files")
            name = names[0]
            z.extractall(path=tmp)
        with open(os.path.join(tmp, name)) as f:
            return json.load(f)


# ---------------------------------------------------------------------- geometry
def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def shot_distance(x: float, y: float) -> float:
    """Feet from the nearer hoop."""
    return min(dist(x, y, hx, hy) for hx, hy in HOOPS)


def split_entities(entities: list) -> tuple[list | None, list[list]]:
    ball, players = None, []
    for e in entities:
        if e[0] == BALL_ID or e[1] == BALL_ID:
            ball = e
        else:
            players.append(e)
    return ball, players


# ------------------------------------------------------------------- extraction
def parse_clock(clock: str) -> float | None:
    """'PT11M22.00S' -> 682.0 seconds remaining in the period."""
    if not isinstance(clock, str) or not clock.startswith("PT"):
        return None
    try:
        body = clock[2:]
        mins, secs = body.split("M")
        return int(mins) * 60 + float(secs.rstrip("S"))
    except (ValueError, AttributeError):
        return None


def find_release(
    moments: list,
    shooter_id: int,
    period: int | None = None,
    clock_s: float | None = None,
    window_s: float = RELEASE_WINDOW_S,
) -> tuple[int, list, str] | None:
    """Locate the shot-release frame. Returns (index, moment, confidence).

    Method (calibrated 2026-07-20 against the NBA's own reported shot distance on a full
    game):

      * The PBP clock alone is NOT reliable — for some plays the tracking stream at the
        stated clock shows the ball at half court. So we search a generous +/- window
        around it rather than trusting the timestamp exactly.
      * Within that window, the release is the frame with the ball at MAXIMUM HEIGHT while
        still within HELD_DIST_FT of the shooter — i.e. the apex of the shooting motion
        just before the ball leaves the hands. This beat both "nearest clock" and
        "closest ball-to-shooter" (corr 0.84 vs 0.61/0.68).

    Confidence:
      "high" - a genuine held-ball frame was found (trustworthy geometry)
      "low"  - no held frame; we fall back to the closest ball-to-shooter frame. These
               should be filtered out of analysis rather than silently trusted.
    """
    held: tuple[float, int, list] | None = None  # (ball z, idx, moment)
    nearest: tuple[float, int, list] | None = None  # (ball-shooter dist, idx, moment)

    for i, m in enumerate(moments):
        if not m or len(m) < 6 or not m[5] or m[2] is None:
            continue
        if period is not None and int(m[0]) != period:
            continue
        if clock_s is not None and abs(float(m[2]) - clock_s) > window_s:
            continue
        ball, players = split_entities(m[5])
        if ball is None:
            continue
        shooter = next((p for p in players if p[1] == shooter_id), None)
        if shooter is None:
            continue
        d = dist(ball[2], ball[3], shooter[2], shooter[3])
        if nearest is None or d < nearest[0]:
            nearest = (d, i, m)
        if d <= HELD_DIST_FT and (held is None or ball[4] > held[0]):
            held = (ball[4], i, m)

    if held is not None:
        return held[1], held[2], "high"
    if nearest is not None:
        return nearest[1], nearest[2], "low"
    return None


def features_at_release(moment: list, shooter_id: int) -> dict | None:
    """Defender distance + context at the release frame."""
    ball, players = split_entities(moment[5])
    shooter = next((p for p in players if p[1] == shooter_id), None)
    if shooter is None:
        return None
    shooter_team = shooter[0]
    defenders = [p for p in players if p[0] != shooter_team]
    if not defenders:
        return None
    dists = sorted(dist(shooter[2], shooter[3], d[2], d[3]) for d in defenders)
    return {
        "quarter": moment[0],
        "game_clock": moment[2],
        "shot_clock": moment[3],
        "shooter_x": round(shooter[2], 3),
        "shooter_y": round(shooter[3], 3),
        "shot_dist_ft": round(shot_distance(shooter[2], shooter[3]), 2),
        "closest_def_dist_ft": round(dists[0], 2),
        "second_def_dist_ft": round(dists[1], 2) if len(dists) > 1 else None,
        "n_defenders_within_6ft": sum(1 for d in dists if d <= 6.0),
        "ball_height_ft": round(ball[4], 2) if ball else None,
    }


def touch_features(moments: list, release_idx: int, shooter_id: int) -> dict:
    """Touch time and dribbles before the shot.

    Touch time  = seconds the ball stayed within 3 ft of the shooter leading into release.
    Dribbles    = count of ball-height local minima during that touch (bounces).
    """
    touch_frames = []
    for i in range(release_idx, -1, -1):
        m = moments[i]
        if not m or len(m) < 6 or not m[5]:
            break
        ball, players = split_entities(m[5])
        shooter = next((p for p in players if p[1] == shooter_id), None)
        if ball is None or shooter is None:
            break
        if dist(ball[2], ball[3], shooter[2], shooter[3]) > HELD_DIST_FT:
            break
        touch_frames.append((m[1], ball[4]))
    if len(touch_frames) < 2:
        return {"touch_time_s": 0.0, "dribbles": 0}

    touch_frames.reverse()
    secs = abs(touch_frames[-1][0] - touch_frames[0][0]) / 1000.0
    heights = [h for _, h in touch_frames]
    dribbles = 0
    for i in range(1, len(heights) - 1):
        if heights[i] < heights[i - 1] and heights[i] <= heights[i + 1] and heights[i] < 4.0:
            dribbles += 1
    return {"touch_time_s": round(secs, 2), "dribbles": dribbles}


def get_pbp_shots(game_id: str) -> pd.DataFrame:
    """Field-goal attempts from play-by-play.

    Uses PlayByPlayV3 — V2 fails to parse for 2015-16 games (KeyError 'resultSet').
    V3 also carries the NBA's own `shotDistance`, which we keep as an independent
    cross-check on our derived geometry.

    `actionNumber` is the join key to the tracking file's `eventId`.
    """
    from nba_api.stats.endpoints import playbyplayv3

    pbp = call(playbyplayv3.PlayByPlayV3, game_id=game_id)
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    shots = pbp[pbp["isFieldGoal"] == 1].copy()
    if shots.empty:
        return pd.DataFrame()
    shots["made"] = (shots["shotResult"].astype(str) == "Made").astype(int)
    shots["is_3pt"] = (pd.to_numeric(shots["shotValue"], errors="coerce") == 3).astype(int)
    shots["clock_s"] = shots["clock"].map(parse_clock)
    shots = shots.rename(
        columns={
            "actionNumber": "EVENTNUM",
            "period": "PERIOD",
            "personId": "PLAYER1_ID",
            "playerName": "PLAYER1_NAME",
            "shotDistance": "nba_shot_dist_ft",
        }
    )
    return shots[
        [
            "EVENTNUM", "PERIOD", "PLAYER1_ID", "PLAYER1_NAME",
            "made", "is_3pt", "nba_shot_dist_ft", "clock_s",
        ]
    ]


def build_moment_index(game: dict) -> dict[int, list]:
    """All moments in the game, de-duplicated by timestamp, bucketed by period.

    We deliberately do NOT join on the tracking `eventId`: that id corresponds to the
    legacy PlayByPlayV2 `EVENTNUM`, whereas PlayByPlayV3 renumbers actions, so the ids
    do not line up and shots get matched to the wrong play. Period + game clock is
    carried by every moment and is unambiguous, so we match on that instead.
    """
    seen: set = set()
    by_period: dict[int, list] = {}
    for ev in game.get("events", []):
        for m in ev.get("moments") or []:
            if not m or len(m) < 6 or not m[5]:
                continue
            ts = m[1]
            if ts in seen:
                continue
            seen.add(ts)
            by_period.setdefault(int(m[0]), []).append(m)
    for p in by_period:
        by_period[p].sort(key=lambda mm: -float(mm[2] if mm[2] is not None else 0))
    return by_period


def extract_game_shots(game: dict, pbp_shots: pd.DataFrame) -> pd.DataFrame:
    """One row per shot with derived tracking features."""
    index = build_moment_index(game)

    rows = []
    for _, s in pbp_shots.iterrows():
        shooter_id = int(s["PLAYER1_ID"])
        period = int(s["PERIOD"]) if pd.notna(s.get("PERIOD")) else None
        clock_s = s.get("clock_s")
        clock_s = float(clock_s) if pd.notna(clock_s) else None
        if period is None or clock_s is None:
            continue
        moments = index.get(period)
        if not moments:
            continue
        found = find_release(moments, shooter_id, period=period, clock_s=clock_s)
        if found is None:
            continue
        idx, moment, release_conf = found
        feats = features_at_release(moment, shooter_id)
        if feats is None:
            continue
        feats.update(touch_features(moments, idx, shooter_id))
        feats["release_confidence"] = release_conf
        feats.update(
            {
                "game_id": game.get("gameid"),
                "game_date": game.get("gamedate"),
                "event_num": int(s["EVENTNUM"]),
                "player_id": shooter_id,
                "player_name": s["PLAYER1_NAME"],
                "made": int(s["made"]),
                "is_3pt": int(s["is_3pt"]),
                "nba_shot_dist_ft": s.get("nba_shot_dist_ft"),
            }
        )
        rows.append(feats)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- demo
def main() -> None:
    demo = "01.01.2016.CHA.at.TOR.7z"
    print("=" * 76)
    print(f"PHASE 1.1a — single-game extraction proof  ({demo})")
    print("=" * 76)

    path = download_game(demo)
    print(f"  archive: {os.path.getsize(path):,} bytes")
    game = load_game(path)
    gid = game["gameid"]
    print(f"  game {gid} on {game['gamedate']} — {len(game['events'])} tracking events")

    pbp = get_pbp_shots(gid)
    print(f"  play-by-play shot events: {len(pbp)}")
    if pbp.empty:
        print("  !! no PBP — cannot join")
        return

    df = extract_game_shots(game, pbp)
    matched = len(df)
    print(f"  shots with tracking features: {matched} / {len(pbp)} ({matched / len(pbp):.0%})")
    if df.empty:
        return

    print("\n  --- derived feature sanity check ---")
    print(f"  closest defender (ft): mean {df['closest_def_dist_ft'].mean():.2f} | "
          f"median {df['closest_def_dist_ft'].median():.2f} | "
          f"min {df['closest_def_dist_ft'].min():.2f} | max {df['closest_def_dist_ft'].max():.2f}")
    print(f"  shot distance   (ft): mean {df['shot_dist_ft'].mean():.2f} | "
          f"median {df['shot_dist_ft'].median():.2f}")
    print(f"  shot clock      (s) : mean {df['shot_clock'].mean():.2f}")
    print(f"  touch time      (s) : mean {df['touch_time_s'].mean():.2f}")
    print(f"  dribbles            : mean {df['dribbles'].mean():.2f}")

    print("\n  --- does defender distance actually predict makes? (the real test) ---")
    bins = pd.cut(df["closest_def_dist_ft"], [0, 2, 4, 6, 100],
                  labels=["0-2 tight", "2-4", "4-6 open", "6+ wide open"])
    tbl = df.groupby(bins, observed=True).agg(shots=("made", "size"), fg_pct=("made", "mean"))
    tbl["fg_pct"] = (tbl["fg_pct"] * 100).round(1)
    print(tbl.to_string())

    print("\n  --- 3PT distance check (should cluster >= 22 ft) ---")
    print(df.groupby("is_3pt")["shot_dist_ft"].agg(["count", "mean", "min"]).round(2).to_string())

    # independent correctness check: our geometry vs the NBA's own reported distance
    chk = df.dropna(subset=["nba_shot_dist_ft"]).copy()
    if not chk.empty:
        chk["nba_shot_dist_ft"] = pd.to_numeric(chk["nba_shot_dist_ft"], errors="coerce")
        chk = chk.dropna(subset=["nba_shot_dist_ft"])
        print("\n  --- our computed shot distance vs NBA's reported (correctness check) ---")
        for label, sub in [
            ("ALL", chk),
            ("high-confidence only", chk[chk["release_confidence"] == "high"]),
        ]:
            if sub.empty:
                continue
            err = (sub["shot_dist_ft"] - sub["nba_shot_dist_ft"]).abs()
            print(f"  {label:22s} n={len(sub):3d} | MAE {err.mean():5.2f} ft | "
                  f"within 3ft {(err <= 3).mean():4.0%} | "
                  f"corr {sub['shot_dist_ft'].corr(sub['nba_shot_dist_ft']):.3f}")
        n_low = (df["release_confidence"] == "low").sum()
        print(f"  release confidence: {len(df) - n_low} high / {n_low} low "
              f"({n_low / len(df):.0%} would be filtered out)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"shots_tracking_{gid}.parquet")
    df.to_parquet(out, index=False)
    print(f"\n  wrote {out}  ({matched} shots x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
