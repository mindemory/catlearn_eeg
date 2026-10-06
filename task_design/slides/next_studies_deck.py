"""Append follow-up-study slides to the edited deck (~/Downloads/2x2 Design Exp0.pptx),
keeping every existing slide as is. Same style and helpers as design_deck.py.

Reads:
  order effects  ~/Documents/data/catlearn_eeg/kernel_model/2x2x2/order_transfer/transfer.json
                 (kernel_model/K03_order_transfer.py)
  4x4 figures    ~/Documents/data/catlearn_eeg/task_design/kernels/{configurations,loadings}/

  python next_studies_deck.py [out.pptx] [slide numbers to preview, e.g. 1,2]
"""

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches

import design_deck as bs
from design_deck import (A_COL, AB_COL, B_COL, DIM, GREY, SCREEN, WHITE, fixation, image, label_in, line, new_slide,
                          pill, screen, shape, text)

DATA = Path.home() / "Documents" / "data" / "catlearn_eeg"
SRC = Path.home() / "Downloads" / "2x2 Design Exp0.pptx"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "2x2 Design Exp0 + next studies.pptx"
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
FIG = DATA / "task_design" / "kernels"
TRANSFER = json.loads((DATA / "kernel_model" / "2x2x2" / "order_transfer" / "transfer.json").read_text())
STUDY_COLORS = ("FFAB40", "A99BF0", "3CC9A0")


def number(slide, n, x, y, d=0.5, color=A_COL):
    c = shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill=color)
    label_in(c, str(n), size=18, color="111111", bold=True)


def study_tag(slide, n):
    """Small 'Study n' marker in the bottom-left corner, so each slide says which study it belongs to."""
    p = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.35, 5.05, 1.1, 0.38, fill=STUDY_COLORS[n - 1], adj=0.5)
    label_in(p, f"Study {n}", size=13, color="111111", bold=True)


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


# ---------------------------------------------------------------- slides

def s_overview(prs):
    s = new_slide(prs, "Next: three online studies")
    items = [("4×4", "more pictures per side"),
             ("Order effects", "does the first rule shape the next?"),
             ("Training & transfer", "people, then AI models")]
    for i, (head, sub) in enumerate(items):
        y = 1.5 + i * 1.15
        number(s, i + 1, 1.2, y, color=STUDY_COLORS[i])
        text(s, 2.0, y - 0.12, 7.0, 0.8, [(head + "\n", {"size": 24}), (sub, {"size": 15, "color": GREY})])


def s_44_display(prs):
    s = new_slide(prs, "4×4: four pictures per side")
    study_tag(s, 1)
    cx, cy, off = screen(s, 2.2, 1.35, 5.6, 2.3, left="fractal34", right="fractal17", item=0.85)
    text(s, cx - off - 0.6, 1.42, 1.2, 0.4, "A", size=20, color=A_COL, align="ctr", bold=True)
    text(s, cx + off - 0.6, 1.42, 1.2, 0.4, "B", size=20, color=B_COL, align="ctr", bold=True)
    for j, k in enumerate(["fractal34", "fractal58", "fractal3", "fractal9"]):
        image(s, k, cx - off - 0.75 + j * 0.5, 4.05, 0.42)
    for j, k in enumerate(["fractal17", "fractal66", "fractal24", "fractal71"]):
        image(s, k, cx + off - 0.75 + j * 0.5, 4.05, 0.42)
    text(s, cx - off - 1.0, 4.35, 2.0, 0.3, "a1 – a4", size=13, color=A_COL, align="ctr")
    text(s, cx + off - 1.0, 4.35, 2.0, 0.3, "b1 – b4", size=13, color=B_COL, align="ctr")
    text(s, 0.8, 4.8, 8.4, 0.4, "16 combinations · 16 rule types · online first, lab if learnable", size=15,
         color=GREY, align="ctr")


def s_44_types(prs):
    s = new_slide(prs, "16 rule types")
    study_tag(s, 1)
    picture(s, FIG / "configurations" / "configs_4x4.png", 2.95, 1.05, h=4.45)
    text(s, 7.55, 2.3, 2.2, 1.6, [("each dot = one combination\n", {}), ("color = the answer", {})], size=13,
         color=GREY)


def s_44_loadings(prs):
    s = new_slide(prs, "What each rule needs")
    study_tag(s, 1)
    picture(s, FIG / "loadings" / "loadings_4x4.png", 0.35, 1.45, w=9.3)
    text(s, 0.8, 4.2, 8.4, 0.8, [("From ", {}), ("A only", {"color": A_COL}), (" (type I) to ", {}),
                                 ("A × B only", {"color": AB_COL}), (" (type XVI)", {})], size=18, align="ctr")


def s_order_design(prs):
    s = new_slide(prs, "Does the first rule shape the second?")
    study_tag(s, 2)
    seq = ["I", "I", "II", "II"]
    cols = [A_COL, A_COL, AB_COL, AB_COL]
    for b, (t, c) in enumerate(zip(seq, cols)):
        x = 1.45 + b * 1.9
        text(s, x, 1.45, 1.4, 0.3, f"block {b + 1}", size=13, color=GREY, align="ctr")
        p = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.85, 1.4, 0.6, fill=c, adj=0.5)
        label_in(p, f"type {t}", size=18, color="111111", bold=True)
        if b < 3:
            line(s, x + 1.48, 2.15, x + 1.82, 2.15, color=GREY, width=1.5, arrow=True)
    text(s, 0.8, 2.8, 8.4, 0.5, "2×2×2 · 6 rule types · new pictures every block · online", size=15, color=GREY,
         align="ctr")
    qs = ["Is type II learned faster or slower after type I?",
          "Does it depend on which type came first?",
          "Can people switch at all once they settle on a rule?"]
    for i, q in enumerate(qs):
        text(s, 1.3, 3.35 + i * 0.5, 7.4, 0.45, q, size=17, align="ctr")


def s_order_model(prs):
    s = new_slide(prs, "Model: a first rule mostly slows the next")
    study_tag(s, 2)
    T = TRANSFER["transfer"]
    types = TRANSFER["types"]
    n = len(types)
    cell, x0, y0 = 0.52, 3.0, 1.55
    vmax = 8.0
    for i in range(n):
        text(s, x0 - 0.55, y0 + i * cell, 0.5, cell, types[i], size=13, color=WHITE, align="r", anchor="ctr")
        text(s, x0 + i * cell, y0 + n * cell + 0.03, cell, 0.3, types[i], size=13, color=WHITE, align="ctr")
        for j in range(n):
            v = 100 * T[i][j]
            a = min(1.0, abs(v) / vmax)
            base = (0x4C, 0xC9, 0xF0) if v < 0 else (0xFF, 0xAB, 0x40)
            col = "".join(f"{int(round(20 + (c - 20) * a)):02X}" for c in base)       # blend from near-black
            c = shape(s, MSO_SHAPE.RECTANGLE, x0 + j * cell, y0 + i * cell, cell, cell, fill=col, line="000000",
                      line_w=1.5)
            label_in(c, "0" if round(v) == 0 else f"{round(v):+d}", size=11, color="111111" if a > 0.45 else WHITE)
    text(s, x0 - 1.45, y0 + n * cell / 2 - 0.4, 0.85, 0.8, "learned\nfirst", size=12, color=GREY, align="ctr",
         anchor="ctr")
    text(s, x0, y0 + n * cell + 0.32, n * cell, 0.3, "learned second", size=12, color=GREY, align="ctr")
    # legend / reading guide on the right
    lx = 6.65
    sw = shape(s, MSO_SHAPE.RECTANGLE, lx, 1.75, 0.35, 0.35, fill="FFAB40")
    text(s, lx + 0.45, 1.72, 2.6, 0.4, "faster than learning it first", size=13, anchor="ctr")
    sw = shape(s, MSO_SHAPE.RECTANGLE, lx, 2.25, 0.35, 0.35, fill="4CC9F0")
    text(s, lx + 0.45, 2.22, 2.6, 0.4, "slower", size=13, anchor="ctr")
    text(s, lx, 2.8, 3.0, 0.6, "numbers: change in accuracy (points)", size=11, color=GREY)
    text(s, lx, 3.4, 3.0, 1.2, "Caveat: the model finds type VI easy; people find it hardest. The data decide.",
         size=12, color=GREY)


def s_order_n(prs):
    s = new_slide(prs, "How many people?")
    study_tag(s, 2)
    rows = [("All orders", "36 orders × 50", "1,800"),
            ("Types I, II, IV, VI", "16 orders × 50", "800"),
            ("Pilot", "4 orders × 50", "200")]
    for i, (name, calc, total) in enumerate(rows):
        y = 1.45 + i * 0.95
        text(s, 1.0, y, 3.2, 0.6, name, size=20, anchor="ctr")
        text(s, 4.2, y, 2.8, 0.6, calc, size=17, color=GREY, anchor="ctr")
        text(s, 7.0, y, 2.0, 0.6, total, size=26, color=A_COL if i == 1 else WHITE, anchor="ctr", align="r",
             bold=True)
    line(s, 1.0, 4.35, 9.0, 4.35, color=DIM, width=1)
    text(s, 0.8, 4.45, 8.4, 0.8,
         "50 per order: 80% power for a medium effect (d ≈ 0.5) vs. learning that type first\n"
         "4 blocks × 64 trials ≈ 20 min online", size=13, color=GREY, align="ctr")


def s_training(prs):
    s = new_slide(prs, "Does training on one design help the other?")
    study_tag(s, 3)
    groups = [("2×2×2", "4×4"), ("4×4", "2×2×2"), (None, "4×4"), (None, "2×2×2")]
    text(s, 2.6, 1.4, 1.8, 0.3, "train", size=13, color=GREY, align="ctr")
    text(s, 5.4, 1.4, 1.8, 0.3, "test", size=13, color=GREY, align="ctr")
    col = {"2×2×2": B_COL, "4×4": AB_COL}
    for i, (train, test) in enumerate(groups):
        y = 1.8 + i * 0.68
        if train:
            p = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 2.6, y, 1.8, 0.5, fill=col[train], adj=0.5)
            label_in(p, train, size=17, color="111111", bold=True)
            line(s, 4.5, y + 0.25, 5.3, y + 0.25, color=GREY, width=1.5, arrow=True)
        else:
            text(s, 2.6, y, 1.8, 0.5, "no training", size=14, color=GREY, align="ctr", anchor="ctr")
        p = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 5.4, y, 1.8, 0.5, fill=col[test], adj=0.5)
        label_in(p, test, size=17, color="111111", bold=True)
    text(s, 0.8, 4.65, 8.4, 0.5, "online · across sessions if needed", size=15, color=GREY, align="ctr")


def s_ai(prs):
    s = new_slide(prs, "If people can't, ask AI models")
    study_tag(s, 3)
    # vision-language model: sees the display
    text(s, 0.6, 1.3, 4.2, 0.4, "Vision-language model", size=18, align="ctr")
    screen(s, 1.0, 1.85, 3.4, 1.7, left="fractal34", right="fractal17", item=0.62)
    text(s, 0.6, 3.7, 4.2, 0.4, "“F or J?”", size=16, color=GREY, align="ctr")
    # language model: random strings instead of pictures
    text(s, 5.2, 1.3, 4.2, 0.4, "Language model", size=18, align="ctr")
    b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 5.5, 1.85, 3.6, 1.7, fill="262626", adj=0.08)
    text(s, 5.7, 1.95, 3.3, 1.5, [("A: ", {"color": A_COL, "bold": True}), ("vexqorlaminth\n", {}),
                                   ("B: ", {"color": B_COL, "bold": True}), ("tuz\n", {}),
                                   ("F or J?", {"color": GREY})], size=17, anchor="ctr")
    text(s, 5.2, 3.7, 4.2, 0.4, "random strings, any length", size=16, color=GREY, align="ctr")
    text(s, 0.8, 4.5, 8.4, 0.5, "Same trials, same feedback, same rule changes", size=17, align="ctr")


def s_memory(prs):
    s = new_slide(prs, "Does memory help or hurt?")
    study_tag(s, 3)
    models = [("Short context", "forgets old trials", DIM),
              ("Full context", "keeps every trial in view", WHITE),
              ("+ test-time memory", "learns as it goes (Titans)", A_COL)]
    for i, (name, sub, col) in enumerate(models):
        x = 0.7 + i * 3.0
        b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.45, 2.6, 1.3, fill=None, line=col, line_w=2, adj=0.1)
        text(s, x + 0.1, 1.55, 2.4, 1.1, [(name + "\n", {"size": 18, "color": col}), (sub, {"size": 13, "color": GREY})],
             align="ctr", anchor="ctr")
    asks = ["How fast does it learn?", "Does it generalize to new pictures?", "Does the old rule get in the way?"]
    for i, q in enumerate(asks):
        text(s, 1.0, 3.1 + i * 0.48, 8.0, 0.45, q, size=17, align="ctr")
    text(s, 0.8, 4.75, 8.4, 0.35, "Behrouz, Zhong & Mirrokni (2025) Titans: Learning to Memorize at Test Time",
         size=11, color=GREY, align="ctr")


SLIDES = [s_overview, s_44_display, s_44_types, s_44_loadings, s_order_design, s_order_model, s_order_n,
          s_training, s_ai, s_memory]


def main():
    bs.make_assets()
    prs = Presentation(str(SRC))
    if ONLY is not None:                      # preview: drop the existing slides
        from pptx.oxml.ns import qn
        ids = prs.slides._sldIdLst
        for sld in list(ids):
            prs.part.drop_rel(sld.get(qn("r:id"))); ids.remove(sld)
    for i, make in enumerate(SLIDES, 1):
        if ONLY is None or str(i) in ONLY.split(","):
            make(prs)
    prs.save(str(OUT))
    print(f"saved {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
