"""Phase 4.2: build the SMOKE one-pager PDF for a basketball-operations audience.

One page, zero equations. The problem in two sentences, one concrete example, one
chart (the Movers view), three reasons to trust the number, and the disclaimer
footer. Navy and blue match the dashboard and the capstone documents.

Run:
    .venv/Scripts/python.exe -m src.report.one_pager
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import fitz
import matplotlib.pyplot as plt
import pandas as pd

from src.pulls._paths import REPO_ROOT
from src.report.figures import nm as name  # one name formatter for every artifact

OUT_DIR = REPO_ROOT / "artifacts"
CHART = OUT_DIR / "movers_chart.png"
PDF = OUT_DIR / "SMOKE_one_pager.pdf"

NAVY = "#1F3864"
BLUE = "#2E75B6"
GRAY = "#6B7280"


def build_chart(n: int = 6) -> None:
    """Diverging bar chart: biggest rank movers, FG% rank versus SMOKE rank."""
    r = pd.read_csv(REPO_ROOT / "data" / "model_outputs" / "rank_table.csv")
    up = r.nlargest(n, "rank_shift_vs_fg")
    down = r.nsmallest(n, "rank_shift_vs_fg").iloc[::-1]
    rows = pd.concat([down, up])
    labels = [name(p) for p in rows["player_name"]]
    vals = rows["rank_shift_vs_fg"].to_numpy()
    colors = [BLUE if v > 0 else GRAY for v in vals]

    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200)
    ax.barh(labels, vals, color=colors, height=0.62)
    ax.axvline(0, color=NAVY, linewidth=0.8)
    ax.set_xlabel("Ranking spots moved, raw FG% rank to SMOKE rank", fontsize=9, color=NAVY)
    ax.tick_params(labelsize=8.5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.set_title(
        "The same players, ranked by raw FG% and then by SMOKE",
        fontsize=10.5, color=NAVY, loc="left", pad=8,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(CHART, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- pdf assembly
PW, PH = 612, 792
ML, MR = 54, 54
CW = PW - ML - MR

FF = {
    "reg": "C:/Windows/Fonts/times.ttf",
    "bold": "C:/Windows/Fonts/timesbd.ttf",
    "ital": "C:/Windows/Fonts/timesi.ttf",
    "sans": "C:/Windows/Fonts/arial.ttf",
    "sansb": "C:/Windows/Fonts/arialbd.ttf",
}
LABEL = {k: f"F{k}" for k in FF}

NAVY_RGB = (0.122, 0.220, 0.392)
BLUE_RGB = (0.180, 0.459, 0.714)
DARK = (0.13, 0.13, 0.13)
GRAY_RGB = (0.42, 0.42, 0.42)


def build_pdf() -> None:
    doc = fitz.open()
    page = doc.new_page(width=PW, height=PH)
    for k, path in FF.items():
        page.insert_font(fontname=LABEL[k], fontfile=path)

    def tb(rect: fitz.Rect, text: str, style: str, size: float, color, lh: float = 1.3):
        return page.insert_textbox(
            rect, text, fontname=LABEL[style], fontfile=FF[style],
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
        "defender distance, shot clock, and touch time, across 128,069 tracked shots. "
        "SMOKE is a player's actual makes minus the makes the model expected. It isolates "
        "shot-making skill from shot selection. Chris Paul led 2014-15 at 53 makes above "
        "expectation, with a confidence range of 25 to 83; his number is skill, not luck.",
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
    bullets = [
        "It repeats. Year-over-year stability is 0.51, higher than effective field goal "
        "percentage at 0.47, and it predicts next season's shot-making nearly twice as "
        "well as efficiency does (0.51 versus 0.28).",
        "It is honest about noise. Every player's value ships with a confidence interval "
        "and a reliability weight, and small samples are shrunk toward average. No "
        "competing shot-quality product publishes either.",
        "It is fair and robust. No playing style is systematically penalized (interior "
        "big men included), and controlling for opponent quality and venue leaves the "
        "rankings essentially unchanged.",
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
        "open and reproducible. Cole Campbell, 2026. Research build.",
        "sans", 8, GRAY_RGB,
    )

    doc.set_metadata({"title": "SMOKE one-pager", "author": "Cole Campbell"})
    doc.subset_fonts(verbose=False)
    doc.save(PDF, deflate=True, garbage=4)
    print(f"wrote {PDF}")


def main() -> None:
    build_chart()
    build_pdf()


if __name__ == "__main__":
    main()
