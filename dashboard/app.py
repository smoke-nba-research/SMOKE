"""SMOKE dashboard (Phase 4.1).

Four pages, information first: Player, Leaderboard, Movers, Methods. Reads the
validation outputs in data/v2/validation/ and the v1 rank table. Keep this simple;
it is an argument, not a product.

Run from the repo root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
try:
    from src.features.names import display_name
except ImportError:
    sys.path.insert(0, str(REPO))
    from src.features.names import display_name

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

# Total games scheduled league-wide in the 2014-15 season (30 teams x 82 / 2); the
# per-shot tracking data covers a subset of these games, read from model_card.json.
SEASON_GAMES_1415 = 1230

_MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _month_day(iso_date: str) -> str:
    """'2014-10-28' -> 'Oct 28'."""
    _, month, day = iso_date.split("-")
    return f"{_MONTHS[int(month) - 1]} {int(day)}"


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


@st.cache_data
def load_players() -> pd.DataFrame:
    """Leaderboard base: error-bar file joined with archetypes and percentile."""
    p = pd.read_csv(VAL / "player_smoke_with_error_bars.csv")
    a = pd.read_csv(VAL / "archetype_fairness.csv")[["player_id", "archetype"]]
    before = len(p)
    p = p.merge(a, on="player_id", how="left")
    assert len(p) == before, "archetype join changed the leaderboard row count"
    p["percentile"] = p["smoke_rate_shrunk"].rank(pct=True)
    p["name"] = p["player"].map(display_name)
    return p


@st.cache_data
def load_ranks() -> pd.DataFrame:
    r = pd.read_csv(V1 / "rank_table.csv")
    r["name"] = r["player_name"].map(display_name)
    return r


@st.cache_data
def load_rank_correlations() -> pd.DataFrame:
    return pd.read_csv(V1 / "rank_correlations.csv")


@st.cache_data
def load_two_season() -> pd.DataFrame:
    return pd.read_parquet(VAL / "stability_2season_players.parquet")


@st.cache_data
def load_model_card() -> dict:
    return json.loads((V1 / "model_card.json").read_text(encoding="utf-8"))


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    return pd.read_csv(V1 / "model_metrics.csv")


@st.cache_data
def load_headline_numbers() -> dict:
    """The manifest written by src.report.numbers; carries the 2015-16 coverage counts."""
    return json.loads((VAL / "headline_numbers.json").read_text(encoding="utf-8"))


def build_methods_table(val: Path = VAL) -> pd.DataFrame:
    """Build the Methods page validation summary table from the validation CSVs.

    A plain function with no Streamlit calls, so it can be exercised directly to
    confirm every headline result formats without error.
    """
    leaderboard = pd.read_csv(val / "player_smoke_with_error_bars.csv")
    split_half = pd.read_csv(val / "split_half_reliability.csv").set_index("design")
    stability = pd.read_csv(val / "stability_correlations.csv")
    convergent = pd.read_csv(val / "convergent_correlations.csv").set_index("metric")
    predictive = pd.read_csv(val / "predictive_summary.csv").set_index("test")["value"]
    confound = pd.read_csv(val / "confound_summary.csv").set_index("stat")["value"]
    arch_means = pd.read_csv(val / "archetype_means.csv")
    arch_omnibus = pd.read_csv(val / "archetype_omnibus.csv").set_index("stat")["value"]

    n_players = len(leaderboard)
    n_distinguishable = int(leaderboard["distinguishable"].sum())
    league_mean = leaderboard["smoke_rate"].mean()
    rel_lo, rel_hi = leaderboard["reliability"].min(), leaderboard["reliability"].max()

    sh_alt = split_half.loc["alternating shots", "spearman_brown"]
    sh_date = split_half.loc["first vs second half by date", "spearman_brown"]

    primary = stability[stability["design"] == "per-season cross-fit"].set_index("metric")
    smoke_stability = primary.loc["SMOKE", "pearson_r"]
    efg_stability = primary.loc["eFG%", "pearson_r"]

    ts_r = convergent.loc["ts_percent", "pearson_r"]
    usg_r = convergent.loc["usg_percent", "pearson_r"]

    smoke_vs_future = predictive["c_corr_smoke_vs_futureSMOKE"]
    pastefg_vs_future = predictive["c_corr_pastEFG_vs_futureSMOKE"]
    p_smoke = predictive["c_p_smoke"]

    r2_context_pct = confound["r2_context"] * 100
    rank_stability = confound["spearman_before_after"]

    interior = arch_means.loc[arch_means["archetype"] == "Interior Finishers"].iloc[0]
    kruskal_p = arch_omnibus["kruskal_p"]
    top = arch_means.loc[arch_means["separable_share"].idxmax()]

    return pd.DataFrame(
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
                f"{n_distinguishable} of {n_players} players separable from average "
                "in one season; error bars ship with every number",
                f"League mean {league_mean:+.1%} (near zero, well calibrated); "
                f"reliability weights range {rel_lo:.2f} to {rel_hi:.2f}",
                f"Split-half (alternating shots) {sh_alt:.2f}; first-half-vs-second-half "
                f"by date {sh_date:.2f}, both after Spearman-Brown correction",
                f"Year over year {smoke_stability:.2f} for SMOKE, versus "
                f"{efg_stability:.2f} for effective field goal percentage "
                "(per-season cross-fit design)",
                f"Correlates with true shooting percentage at {ts_r:.2f}, far more than "
                f"with usage rate at {usg_r:.2f} -- shot-making, not shot volume or "
                "role, as predicted",
                f"Predicts next-season SMOKE at {smoke_vs_future:.2f}, versus "
                f"{pastefg_vs_future:.2f} for past effective field goal percentage "
                f"({_fmt_p(p_smoke)} on the SMOKE term)",
                f"Opponent quality and venue explain {r2_context_pct:.1f}% of variance; "
                f"player ranking holds at {rank_stability:.3f} before versus after "
                "adjustment",
                f"Interior Finishers, the most shot-selection-constrained group, are not "
                f"penalized (p={interior['p_weighted_vs_0']:.2f} vs. zero); no significant "
                f"difference across archetypes (kruskal p={kruskal_p:.2f}); "
                f"{top['archetype']} are the best-measured group "
                f"({top['separable_share']:.0%} separable)",
            ],
        }
    )


def build_model_quality_line(v1: Path = V1) -> str:
    """One-line model scorecard: boosted model against the base-rate row."""
    metrics = pd.read_csv(v1 / "model_metrics.csv").set_index("model")
    base = metrics.loc["Base rate"]
    boosted = metrics.loc["Gradient-Boosted Trees"]
    return (
        f"Model quality: the gradient-boosted model reaches {boosted['roc_auc']:.3f} AUC, "
        f"{boosted['log_loss']:.3f} log loss, and {boosted['brier']:.3f} Brier score on "
        f"held-out shots, against {base['roc_auc']:.3f} AUC, {base['log_loss']:.3f} log "
        f"loss, and {base['brier']:.3f} Brier for a model that always predicts the base rate."
    )


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

    st.divider()
    st.markdown("#### Two-season stability model")
    st.caption(
        "These values come from a separate two-season model: four difficulty features "
        "shared across seasons, cross-fitted per season, unshrunk, fit on partial "
        "seasons -- built for the year-over-year stability test. They are on a "
        "different scale from the leaderboard card above; the two are not "
        "directly comparable."
    )
    t = two[two["player_id"] == row["player_id"]]
    if len(t):
        r = t.iloc[0]
        st.dataframe(
            pd.DataFrame(
                {
                    "Season": ["2014-15", "2015-16"],
                    "SMOKE rate (two-season model)": [
                        f"{r['smoke_14']:+.1%}",
                        f"{r['smoke_16']:+.1%}",
                    ],
                    "Shots": [int(r["shots_14"]), int(r["shots_16"])],
                }
            ),
            hide_index=True,
        )
    else:
        st.caption("Not in the two-season stability panel (insufficient shots in one season).")


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


def page_movers(ranks: pd.DataFrame, rank_correlations: pd.DataFrame) -> None:
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
                    "SMOKE rank": up["smoke_rank"].astype(int),
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
                    "SMOKE rank": down["smoke_rank"].astype(int),
                    "Moved": down["rank_shift_vs_fg"].map(lambda v: f"{v:.0f}"),
                }
            ),
            hide_index=True,
        )
    fg_rho = rank_correlations.loc[
        rank_correlations["comparison"] == "smoke_rate vs fg_pct", "spearman_rho"
    ].iloc[0]
    st.write(
        "Both lists are explainable, and that is the argument: SMOKE credits shot-making "
        "and withholds credit for shot selection. Rank correlation with raw field goal "
        f"percentage is {fg_rho:.2f}, so this is new information, not a repackage."
    )


def page_methods() -> None:
    st.subheader("Methods, in plain language")
    card = load_model_card()
    start, end = card["date_range"]
    st.write(
        "A gradient-boosted model estimates each shot's make probability from its "
        "context: distance, defender distance, shot clock, touch time, and related "
        f"features, trained on {card['n_shots']:,} shots from {card['n_games']} of "
        f"{SEASON_GAMES_1415} games in the 2014-15 season ({_month_day(start)} to "
        f"{_month_day(end)}), the season with public per-shot tracking detail. "
        "Expectations are cross-fitted: each shot is scored by a model that never saw "
        "its game, so residuals are not flattered by fitting noise. SMOKE is actual "
        "makes minus expected makes, aggregated per player, then shrunk toward league "
        "average in proportion to noise."
    )
    st.markdown("**Validation summary.** Eight tests, all reproducible from the repository.")
    st.dataframe(build_methods_table(), hide_index=True)
    st.caption(build_model_quality_line())
    cov = load_headline_numbers()["data"]
    st.markdown(
        "**Limitations, stated plainly.** SMOKE measures shot-making only; it says "
        "nothing about defense, playmaking, or rebounding. Per-shot difficulty data is "
        f"public for two partial seasons: 2014-15 ({card['n_games']} of {SEASON_GAMES_1415} games) "
        f"and 2015-16 ({cov['n_games_1516']} games, {cov['n_games_1516'] / SEASON_GAMES_1415:.0%} of the "
        "season), the latter via a preserved tracking archive; claims rest on the former. "
        "One season rarely separates an individual player from average, which is why "
        "intervals and shrinkage ship with every number."
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
    rank_correlations = load_rank_correlations()
    with tab1:
        page_player(players, two)
    with tab2:
        page_leaderboard(players)
    with tab3:
        page_movers(ranks, rank_correlations)
    with tab4:
        page_methods()
    footer()


main()
