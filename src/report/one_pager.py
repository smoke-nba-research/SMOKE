"""Phase 4.2: build the SMOKE one-pager PDF for a basketball-operations audience.

One page, zero equations. The problem in two sentences, one concrete example, one
chart (the Movers view), three reasons to trust the number, and the disclaimer
footer. Navy and blue match the dashboard and the capstone documents.

Run:
    .venv/Scripts/python.exe -m src.report.one_pager
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pandas as pd

from src.models.build_model_outputs import load_scored_shots
from src.pulls._paths import REPO_ROOT
from src.report import figures
from src.report.figures import nm as name  # one name formatter for every artifact

OUT_DIR = REPO_ROOT / "artifacts"
VAL = REPO_ROOT / "data" / "v2" / "validation"
CHART = OUT_DIR / "one_pager_movers.png"
PDF = OUT_DIR / "SMOKE_one_pager.pdf"


# ---------------------------------------------------------------- pdf assembly
PW, PH = 612, 792
ML, MR = 54, 54
CW = PW - ML - MR

# Windows font paths, tried first; each falls back to a PyMuPDF base-14 font (no
# file needed) so the script also runs on macOS/Linux.
FF = {
    "reg": "C:/Windows/Fonts/times.ttf",
    "bold": "C:/Windows/Fonts/timesbd.ttf",
    "ital": "C:/Windows/Fonts/timesi.ttf",
    "sans": "C:/Windows/Fonts/arial.ttf",
    "sansb": "C:/Windows/Fonts/arialbd.ttf",
}
BASE14 = {
    "reg": "Times-Roman",
    "bold": "Times-Bold",
    "ital": "Times-Italic",
    "sans": "Helvetica",
    "sansb": "Helvetica-Bold",
}
LABEL = {k: f"F{k}" for k in FF}

NAVY_RGB = (0.122, 0.220, 0.392)
BLUE_RGB = (0.180, 0.459, 0.714)
DARK = (0.13, 0.13, 0.13)
GRAY_RGB = (0.42, 0.42, 0.42)


def load_numbers() -> dict:
    """Every prose number in the one-pager, read from the validation outputs."""
    players = pd.read_csv(VAL / "player_smoke_with_error_bars.csv")
    top = players.loc[players["smoke_total_shrunk"].idxmax()]

    n_shots = len(load_scored_shots())

    stability = pd.read_csv(VAL / "stability_correlations.csv")
    stability = stability[stability["design"] == "per-season cross-fit"]
    r_smoke = float(stability.loc[stability["metric"] == "SMOKE", "pearson_r"].iloc[0])
    r_efg = float(stability.loc[stability["metric"] == "eFG%", "pearson_r"].iloc[0])

    predictive = pd.read_csv(VAL / "predictive_summary.csv").set_index("test")["value"]
    smoke_pred = float(predictive["c_corr_smoke_vs_futureSMOKE"])
    efg_pred = float(predictive["c_corr_pastEFG_vs_futureSMOKE"])

    confound = pd.read_csv(VAL / "confound_summary.csv").set_index("stat")["value"]
    r2_context = float(confound["r2_context"])
    spearman_before_after = float(confound["spearman_before_after"])

    return {
        "top_name": name(top["player"]),
        "top_raw": float(top["smoke_total"]),
        "top_lo": float(top["smoke_total_lo"]),
        "top_hi": float(top["smoke_total_hi"]),
        "n_shots": n_shots,
        "r_smoke": r_smoke,
        "r_efg": r_efg,
        "smoke_pred": smoke_pred,
        "efg_pred": efg_pred,
        "r2_context": r2_context,
        "spearman_before_after": spearman_before_after,
    }


def build_pdf() -> None:
    n = load_numbers()

    doc = fitz.open()
    page = doc.new_page(width=PW, height=PH)
    font_name: dict[str, str] = {}
    font_file: dict[str, str | None] = {}
    for k, path in FF.items():
        if Path(path).exists():
            page.insert_font(fontname=LABEL[k], fontfile=path)
            font_name[k], font_file[k] = LABEL[k], path
        else:
            font_name[k], font_file[k] = BASE14[k], None

    def tb(rect: fitz.Rect, text: str, style: str, size: float, color, lh: float = 1.3):
        return page.insert_textbox(
            rect, text, fontname=font_name[style], fontfile=font_file[style],
            fontsize=size, color=color, lineheight=lh,
        )

    def block(text: str, style: str, size: float, color, y: float, gap: float = 8) -> float:
        rect = fitz.Rect(ML, y, ML + CW, PH - 40)
        leftover = tb(rect, text, style, size, color)
        return y + (rect.height - leftover) + gap

    y = 46.0
    y = block("SMOKE", "bold", 30, NAVY_RGB, y, gap=0)
    y = block(
        "Shots Made Over Known Expectation. Shot-making, separated from shot selection.",
        "ital", 12, BLUE_RGB, y, gap=12,
    )
    page.draw_line(fitz.Point(ML, y), fitz.Point(PW - MR, y), color=BLUE_RGB, width=1)
    y += 14

    y = block("The problem", "sansb", 11.5, NAVY_RGB, y, gap=3)
    y = block(
        "Field goal percentage rewards two different things at once: taking easy shots "
        "and making hard ones. Two players can post the same percentage while one lives "
        "on open dunks and the other makes contested jumpers, and the box score cannot "
        "tell them apart.",
        "reg", 11, DARK, y, gap=10,
    )

    y = block("What SMOKE measures", "sansb", 11.5, NAVY_RGB, y, gap=3)
    y = block(
        "A model estimates every shot's make probability from its difficulty: distance, "
        f"defender distance, shot clock, and touch time, across {n['n_shots']:,} tracked shots. "
        "SMOKE is a player's actual makes minus the makes the model expected. It isolates "
        f"shot-making skill from shot selection. {n['top_name']} led 2014-15 at {n['top_raw']:.0f} "
        f"makes above expectation, with a confidence range of {n['top_lo']:.0f} to {n['top_hi']:.0f}; "
        "his number is skill, not luck.",
        "reg", 11, DARK, y, gap=12,
    )

    # chart
    img_h = CW * 0.46
    page.insert_image(fitz.Rect(ML, y, ML + CW, y + img_h), filename=str(CHART))
    y += img_h + 4
    y = block(
        "Risers make difficult shots that raw percentages undervalue. Fallers post high "
        "percentages on shots the model expected them to make anyway. Both lists are "
        "explainable, which is the point.",
        "ital", 9.5, GRAY_RGB, y, gap=12,
    )

    y = block("Why trust this number", "sansb", 11.5, NAVY_RGB, y, gap=4)
    stability_word = "higher than" if n["r_smoke"] >= n["r_efg"] else "lower than"
    predictive_word = "better than" if n["smoke_pred"] >= n["efg_pred"] else "about as well as"
    bullets = [
        f"It repeats. Year-over-year stability is {n['r_smoke']:.2f}, {stability_word} effective "
        f"field goal percentage at {n['r_efg']:.2f}, and it predicts next season's shot-making "
        f"{predictive_word} efficiency does ({n['smoke_pred']:.2f} versus {n['efg_pred']:.2f}).",
        "It is honest about noise. Every player's value ships with a confidence interval "
        "and a reliability weight, and small samples are shrunk toward average. No "
        "competing shot-quality product publishes either.",
        "It is fair and robust. No playing style is structurally penalized (interior "
        f"big men included). Opponent quality and venue explain just {n['r2_context']:.1%} of the "
        "variance in SMOKE, and rankings before and after controlling for them correlate at "
        f"a Spearman correlation of {n['spearman_before_after']:.3f}.",
    ]
    for b in bullets:
        rect = fitz.Rect(ML + 14, y, ML + CW, PH - 40)
        leftover = tb(rect, b, "reg", 10.5, DARK)
        page.draw_circle(fitz.Point(ML + 4, y + 5.2), 1.8, color=BLUE_RGB, fill=BLUE_RGB)
        y += (rect.height - leftover) + 5

    # footer
    page.draw_line(fitz.Point(ML, PH - 64), fitz.Point(PW - MR, PH - 64), color=BLUE_RGB, width=0.8)
    tb(
        fitz.Rect(ML, PH - 58, ML + CW, PH - 22),
        "SMOKE is an independent analysis of publicly available NBA data and is not "
        "affiliated with or endorsed by the NBA. Methods, validation, and code are "
        "open and reproducible. Campbell, Rajesh, and Wyllie, 2026. Research build.",
        "sans", 8, GRAY_RGB,
    )

    doc.set_metadata({"title": "SMOKE one-pager", "author": "Cole Campbell, Marc Rajesh, Calder Wyllie"})
    doc.subset_fonts(verbose=False)
    doc.save(PDF, deflate=True, garbage=4)
    print(f"wrote {PDF}")


def main() -> None:
    figures.fig_movers(
        n=6, filename="one_pager_movers.png",
        title="The same players, ranked by raw FG% and then by SMOKE",
    )
    build_pdf()


if __name__ == "__main__":
    main()
