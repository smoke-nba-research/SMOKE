"""SMOKE dashboard (Phase 4.1).

Four pages, information first: Player, Leaderboard, Movers, Methods. Reads the
validation outputs in data/v2/validation/ and the v1 rank table. Keep this simple;
it is an argument, not a product.

Run from the repo root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
VAL = REPO / "data" / "v2" / "validation"
V1 = REPO / "data" / "model_outputs"

NAVY = "#1F3864"
BLUE = "#2E75B6"

TAGLINE = "Shot-making, separated from shot selection."
FOOTER = (
    "SMOKE (Shots Made Over Known Expectation) is an independent analysis of "
    "publicly available NBA data. It is not affiliated with or endorsed by the NBA. "
    "Season shown: 2014-15. Research build."
)

# display fixups for names that .title() gets wrong
DISPLAY_FIX = {
    "lebron james": "LeBron James",
    "demarcus cousins": "DeMarcus Cousins",
    "deandre jordan": "DeAndre Jordan",
    "demarre carroll": "DeMarre Carroll",
    "jj barea": "J.J. Barea",
    "cj mccollum": "C.J. McCollum",
    "cj miles": "C.J. Miles",
    "kyle oquinn": "Kyle O'Quinn",
    "amare stoudemire": "Amar'e Stoudemire",
    "dirk nowtizski": "Dirk Nowitzki",
}


def display_name(raw: str) -> str:
    return DISPLAY_FIX.get(raw, raw.title())


@st.cache_data
def load_players() -> pd.DataFrame:
    """Leaderboard base: error-bar file joined with archetypes and percentile."""
    p = pd.read_csv(VAL / "player_smoke_with_error_bars.csv")
    a = pd.read_csv(VAL / "archetype_fairness.csv")[["player_name", "archetype"]]
    p = p.merge(a, left_on="player", right_on="player_name", how="left").drop(
        columns=["player_name"]
    )
    p["percentile"] = p["smoke_rate_shrunk"].rank(pct=True)
    p["name"] = p["player"].map(display_name)
    return p


@st.cache_data
def load_ranks() -> pd.DataFrame:
    r = pd.read_csv(V1 / "rank_table.csv")
    r["name"] = r["player_name"].map(display_name)
    return r


@st.cache_data
def load_two_season() -> pd.DataFrame:
    return pd.read_parquet(VAL / "stability_2season_players.parquet")


def header() -> None:
    st.markdown(
        f"<h1 style='color:{NAVY}; margin-bottom:0'>SMOKE</h1>"
        f"<p style='color:{BLUE}; font-size:1.1rem; margin-top:0'>"
        f"Shots Made Over Known Expectation. {TAGLINE}</p>",
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.divider()
    st.caption(FOOTER)


def page_player(players: pd.DataFrame, two: pd.DataFrame) -> None:
    st.subheader("Player lookup")
    st.write(
        "SMOKE is a player's actual field goal percentage minus the percentage a "
        "shot-difficulty model expected, given where and how they shot. Positive means "
        "they make more than their shot diet predicts."
    )
    pick = st.selectbox(
        "Player (2014-15, minimum 150 shots)",
        options=players.sort_values("name")["name"].tolist(),
        index=None,
        placeholder="Choose a player",
    )
    if not pick:
        return
    row = players[players["name"] == pick].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SMOKE rate (shrunk)", f"{row['smoke_rate_shrunk']:+.1%}")
    c2.metric("Makes above expected", f"{row['smoke_total']:+.0f}")
    c3.metric("Percentile", f"{row['percentile']:.0%}")
    c4.metric("Shots", f"{row['shots']:,.0f}")

    st.write(
        f"95% interval on makes above expected: "
        f"[{row['smoke_total_lo']:+.0f}, {row['smoke_total_hi']:+.0f}]. "
        f"Reliability weight {row['reliability']:.2f} "
        f"(how much of this number is repeatable skill rather than sampling variation)."
    )
    if bool(row["distinguishable"]):
        st.success("Statistically distinguishable from league average this season.")
    else:
        st.info(
            "Not statistically distinguishable from league average this season. "
            "Most single-season numbers are not; that is honest, not a defect."
        )
    if isinstance(row.get("archetype"), str):
        st.write(f"Playing-style archetype: **{row['archetype']}**.")

    t = two[two["player"] == row["player"]]
    if len(t):
        r = t.iloc[0]
        st.write("Two-season view (2015-16 figures come from a half-season of tracking):")
        st.dataframe(
            pd.DataFrame(
                {
                    "Season": ["2014-15", "2015-16"],
                    "SMOKE rate": [f"{r['smoke_14']:+.1%}", f"{r['smoke_16']:+.1%}"],
                    "Shots": [int(r["shots_14"]), int(r["shots_16"])],
                }
            ),
            hide_index=True,
        )


def page_leaderboard(players: pd.DataFrame) -> None:
    st.subheader("Leaderboard")
    st.write(
        "Shrunk values lead: small samples are pulled toward league average in "
        "proportion to their noise, so a 200-shot hot streak cannot top the board. "
        "The interval column is the honest range on each player's raw total."
    )
    min_shots = st.slider("Minimum shots", 150, 800, 300, step=50)
    view = players[players["shots"] >= min_shots].sort_values(
        "smoke_total_shrunk", ascending=False
    )
    table = pd.DataFrame(
        {
            "Player": view["name"],
            "Archetype": view["archetype"].fillna(""),
            "Shots": view["shots"].astype(int),
            "SMOKE rate (shrunk)": view["smoke_rate_shrunk"].map(lambda v: f"{v:+.1%}"),
            "Makes above expected (raw)": view["smoke_total"].map(lambda v: f"{v:+.0f}"),
            "95% interval": [
                f"[{lo:+.0f}, {hi:+.0f}]"
                for lo, hi in zip(view["smoke_total_lo"], view["smoke_total_hi"], strict=True)
            ],
            "Separable from average": view["distinguishable"].map({True: "Yes", False: "No"}),
        }
    )
    st.dataframe(table, hide_index=True, height=520)
    st.caption(
        f"{len(view)} players at this floor. "
        f"{(view['distinguishable']).mean():.0%} are statistically separable from average."
    )


def page_movers(ranks: pd.DataFrame) -> None:
    st.subheader("Movers: where SMOKE disagrees with the box score")
    st.write(
        "Rank every qualified 2014-15 player two ways: by raw field goal percentage, "
        "and by SMOKE. The players who move furthest are the point of the metric. "
        "Risers make hard shots that raw percentages undervalue. Fallers post high "
        "percentages on easy shots that the model expected them to make anyway."
    )
    n = st.slider("Show top movers", 5, 20, 10)
    up = ranks.nlargest(n, "rank_shift_vs_fg")
    down = ranks.nsmallest(n, "rank_shift_vs_fg")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<h4 style='color:{NAVY}'>Risers</h4>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "Player": up["name"],
                    "FG% rank": up["fg_rank"].astype(int),
                    "SMOKE rank": up["adjusted_rank"].astype(int),
                    "Moved": up["rank_shift_vs_fg"].map(lambda v: f"+{v:.0f}"),
                }
            ),
            hide_index=True,
        )
    with c2:
        st.markdown(f"<h4 style='color:{NAVY}'>Fallers</h4>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "Player": down["name"],
                    "FG% rank": down["fg_rank"].astype(int),
                    "SMOKE rank": down["adjusted_rank"].astype(int),
                    "Moved": down["rank_shift_vs_fg"].map(lambda v: f"{v:.0f}"),
                }
            ),
            hide_index=True,
        )
    st.write(
        "Both lists are explainable, and that is the argument: SMOKE credits shot-making "
        "and withholds credit for shot selection. Rank correlation with raw field goal "
        "percentage is 0.39, so this is new information, not a repackage."
    )


def page_methods() -> None:
    st.subheader("Methods, in plain language")
    st.write(
        "A gradient-boosted model estimates each shot's make probability from its "
        "context: distance, defender distance, shot clock, touch time, and related "
        "features, trained on 128,069 shots from 2014-15, the season with public "
        "per-shot tracking detail. SMOKE is actual makes minus expected makes, "
        "aggregated per player, then shrunk toward league average in proportion to noise."
    )
    st.markdown("**Validation summary.** Eight tests, all reproducible from the repository.")
    st.dataframe(
        pd.DataFrame(
            {
                "Test": [
                    "Error bars",
                    "Shrinkage",
                    "Reliability, within season",
                    "Stability, across seasons",
                    "Convergent validity",
                    "Predictive validity",
                    "Confound checks",
                    "Archetype fairness",
                ],
                "Headline result": [
                    "49 of 266 players separable from average in one season; the top tier separates cleanly",
                    "League mean near zero (well calibrated); low-volume outliers regress",
                    "Split-half 0.57 after Spearman-Brown correction",
                    "Year over year 0.51, above effective field goal percentage at 0.47",
                    "Correlations land in the predicted order; every large disagreement is explainable",
                    "Predicts next-season shot-making at 0.51 versus 0.28 for efficiency",
                    "Opponent quality and venue explain 2.2% of variance; ranking holds at 0.99",
                    "No playing style penalized; interior big men are the best-measured group",
                ],
            }
        ),
        hide_index=True,
    )
    st.markdown(
        "**Limitations, stated plainly.** SMOKE measures shot-making only; it says "
        "nothing about defense, playmaking, or rebounding. Per-shot difficulty data is "
        "public for 2014-15 and, through a preserved tracking archive, a half of "
        "2015-16; claims rest on the former. One season rarely separates an individual "
        "player from average, which is why intervals and shrinkage ship with every number."
    )
    st.markdown(
        "**Full detail.** The validation dossier (VALIDATION.md) and all code live in "
        "the project repository. Every figure on this dashboard regenerates from "
        "public data."
    )


def main() -> None:
    st.set_page_config(page_title="SMOKE", page_icon=None, layout="wide")
    header()
    tab1, tab2, tab3, tab4 = st.tabs(["Player", "Leaderboard", "Movers", "Methods"])
    players = load_players()
    ranks = load_ranks()
    two = load_two_season()
    with tab1:
        page_player(players, two)
    with tab2:
        page_leaderboard(players)
    with tab3:
        page_movers(ranks)
    with tab4:
        page_methods()
    footer()


main()
