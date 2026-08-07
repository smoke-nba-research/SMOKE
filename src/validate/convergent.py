"""Phase 2.5 — convergent validity: does SMOKE agree with established all-in-one metrics?

The test a team's staff runs to decide whether a new metric is (a) measuring something real, not noise
[too-low correlation = noise], and (b) not just a redundant repackage of something they
already have [too-high correlation = pointless]. The *disagreements* are the actual finding:
where SMOKE and an all-in-one diverge, we should be able to explain why in one sentence —
usually "the all-in-one credits defense/playmaking that SMOKE, by design, ignores."

Comparison metrics (2014-15, Basketball-Reference advanced stats via the sumitrodatta
Kaggle set — fully public, IP-safe, and the canonical academic benchmark):
  * BPM  — total box plus/minus (offense + defense + everything)
  * OBPM — offensive box plus/minus (the most relevant single comparator)
  * PER  — player efficiency rating
  * TS%  — true shooting (raw efficiency, NOT difficulty-adjusted — the thing SMOKE improves on)
  * USG% — usage rate (a negative control: shot-making skill should NOT track how often you shoot)

Note on DARKO: the roadmap named DARKO DPM as the lead comparator, but darko.app's free
download is a current-season snapshot with no clean historical endpoint (checked 2026-07-21),
so it can't supply 2014-15 values. BPM/OBPM are fully public for every season and are the
more standard academic benchmark anyway — used here instead, noted honestly.

Run:
    .venv/Scripts/python.exe -m src.validate.convergent
"""

from __future__ import annotations

import re
import sys
import unicodedata

import pandas as pd
from scipy import stats

from src.pulls._paths import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL_DIR = REPO_ROOT / "data" / "v2" / "validation"
SMOKE_CSV = VAL_DIR / "player_smoke_with_error_bars.csv"   # from reliability.py (2.1-2.3)
ADVANCED = REPO_ROOT / "data" / "raw" / "kaggle" / "historical_stats" / "Advanced.csv"
SEASON = 2015  # 2014-15

# manual bridges for names the normalizer can't reconcile:
#   Kaggle misspellings, and real cross-source name differences (a legal name change).
NAME_FIXES = {
    "dirk nowtizski": "dirk nowitzki",       # Kaggle misspelling
    "danilo gallinai": "danilo gallinari",
    "mnta ellis": "monta ellis",
    "jimmer dredette": "jimmer fredette",
    "dwayne wade": "dwyane wade",            # Kaggle misspelling of Dwyane
    "steve adams": "steven adams",
    "time hardaway": "tim hardaway",
    "nerles noel": "nerlens noel",
    "beno urdih": "beno udrih",
    "jon ingles": "joe ingles",
    "jose juan barea": "jj barea",
    "jose barea": "jj barea",
    "nene hilario": "nene",                  # Advanced lists him mononymously
    "enes kanter": "enes freedom",           # legal name change; datasets disagree
}

# letters that NFKD does NOT fold to ASCII (base letters, not accented forms)
CHAR_MAP = str.maketrans({"ı": "i", "ø": "o", "đ": "d", "ł": "l", "ß": "ss"})


def norm_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes so two sources' names align."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip().translate(CHAR_MAP)    # ömer aşık -> omer asik
    s = re.sub(r"[.'`]", "", s)                  # cj mccollum, dangelo russell
    s = re.sub(r"[-]", " ", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NAME_FIXES.get(s, s)


def load_smoke() -> pd.DataFrame:
    if not SMOKE_CSV.exists():
        raise FileNotFoundError(
            f"{SMOKE_CSV} not found — run `python -m src.validate.reliability` first (2.1-2.3)."
        )
    s = pd.read_csv(SMOKE_CSV)
    s["key"] = s["player"].map(norm_name)
    return s


def load_advanced() -> pd.DataFrame:
    a = pd.read_csv(ADVANCED)
    a = a[a["season"] == SEASON].copy()
    # a player traded mid-season has multiple rows; keep the one with the most minutes
    a = a.sort_values("mp", ascending=False).drop_duplicates("player_id", keep="first")
    a["key"] = a["player"].map(norm_name)
    cols = ["key", "player", "bpm", "obpm", "dbpm", "per", "ts_percent", "usg_percent", "mp", "g"]
    return a[cols]


def main() -> None:
    smoke = load_smoke()
    adv = load_advanced()
    print("=" * 78)
    print(f"PHASE 2.5 — convergent validity ({SEASON - 1}-{str(SEASON)[-2:]})")
    print("=" * 78)
    print(f"  SMOKE players: {len(smoke)}  |  advanced-stats players: {len(adv)}")

    m = smoke.merge(adv, on="key", how="inner", suffixes=("", "_adv"))
    print(f"  matched on normalized name: {len(m)}  "
          f"({len(m) / len(smoke):.0%} of SMOKE players)")
    unmatched = sorted(set(smoke["key"]) - set(adv["key"]))
    if unmatched:
        print(f"  unmatched ({len(unmatched)}): {', '.join(unmatched[:12])}"
              f"{' ...' if len(unmatched) > 12 else ''}")

    # SMOKE point estimate to correlate against: prefer shrunk rate, fall back to raw
    smoke_col = "smoke_rate_shrunk" if "smoke_rate_shrunk" in m.columns else "smoke_rate"
    print(f"\n  correlating {smoke_col} against each all-in-one metric:")
    print("  " + "-" * 62)
    print(f"  {'metric':10s} {'pearson_r':>10} {'spearman':>10}   expectation")
    print("  " + "-" * 62)
    expect = {
        "obpm": "moderate + (offense incl. shooting)",
        "bpm": "lower + (includes defense)",
        "per": "moderate + (efficiency-driven)",
        "ts_percent": "moderate + (raw effic. SMOKE refines)",
        "usg_percent": "~0 (negative control)",
    }
    rows = []
    for col in ["obpm", "bpm", "per", "ts_percent", "usg_percent"]:
        sub = m[[smoke_col, col]].dropna()
        r = float(stats.pearsonr(sub[smoke_col], sub[col]).statistic)
        rho = float(stats.spearmanr(sub[smoke_col], sub[col]).statistic)
        rows.append({"metric": col, "pearson_r": round(r, 3), "spearman": round(rho, 3), "n": len(sub)})
        print(f"  {col:10s} {r:>+10.3f} {rho:>+10.3f}   {expect[col]}")
    print("  " + "-" * 62)

    # the disagreement analysis — the actual paper section
    # rank both by percentile so they're comparable, look at the biggest gaps
    m2 = m.dropna(subset=[smoke_col, "obpm"]).copy()
    m2["smoke_pctl"] = m2[smoke_col].rank(pct=True)
    m2["obpm_pctl"] = m2["obpm"].rank(pct=True)
    m2["gap"] = m2["smoke_pctl"] - m2["obpm_pctl"]  # + = SMOKE likes them more than OBPM does

    print("\n  BIGGEST DISAGREEMENTS vs OBPM (the finding — explain each in one line):")
    print("\n  SMOKE rates them MUCH HIGHER than OBPM does")
    print("  (elite shot-makers whose all-around box impact is modest — often pure shooters):")
    for _, r in m2.sort_values("gap", ascending=False).head(8).iterrows():
        print(f"    {r['player'][:22]:22s}  SMOKE pctl {r['smoke_pctl']:.0%}  vs OBPM pctl {r['obpm_pctl']:.0%}   (obpm {r['obpm']:+.1f})")
    print("\n  OBPM rates them MUCH HIGHER than SMOKE does")
    print("  (offensive value from playmaking/volume/drawing fouls, not shot-making over expectation):")
    for _, r in m2.sort_values("gap").head(8).iterrows():
        print(f"    {r['player'][:22]:22s}  SMOKE pctl {r['smoke_pctl']:.0%}  vs OBPM pctl {r['obpm_pctl']:.0%}   (obpm {r['obpm']:+.1f})")

    m2.sort_values("gap", ascending=False).to_csv(VAL_DIR / "convergent_validity.csv", index=False)
    pd.DataFrame(rows).to_csv(VAL_DIR / "convergent_correlations.csv", index=False)
    print(f"\n  wrote {VAL_DIR / 'convergent_correlations.csv'}")
    print(f"  wrote {VAL_DIR / 'convergent_validity.csv'}  ({len(m2)} matched players)")


if __name__ == "__main__":
    main()
