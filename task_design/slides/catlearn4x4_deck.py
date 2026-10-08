"""4x4 category-learning study deck (catlearn_4x4_prolific 1.0.0), in the style of
~/Downloads/2x2 Design Exp0 + next studies (1).pptx.

Describes the online task as it is now: F/J categorisation of fractal pairs with feedback,
practice (2x2), early screening, then three 4x4 test blocks VI -> X -> II, and a bonus per test block. One idea per
slide, little text. Timing, keys, practice criterion, rules, sequences, bonus and calibration
are read from the task's own src/config.js, and the fractals are the task's own (the vivid set,
fractal_stimuli/), so the slides match what participants see.

Keeps the template's master, layouts and font, drops its slides, and draws everything as
editable shapes with the real fractal images. Replaces slot4x4_deck.py (the slot-machine
version, kept for its deck).

  python catlearn4x4_deck.py [out.pptx] [slide numbers to build, e.g. 2,3]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Pt

import design_deck as dd
from design_deck import A_COL, B_COL, GREY, SCREEN, WHITE, image, label_in, line, shape, text

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "catlearn_4x4_prolific"
CONFIG = TASK / "src" / "config.js"
GROUP_DIR = TASK / "stimuli" / "fractal_groups"
DATA = Path.home() / "Documents" / "data" / "catlearn_eeg"
POOL_VIVID, POOL_OLD = DATA / "fractal_pool_vivid", DATA / "fractal_pool"
TEMPLATE = Path.home() / "Downloads" / "2x2 Design Exp0 + next studies (1).pptx"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "4x4 Category Learning Study.pptx"
ONLY = sys.argv[2] if len(sys.argv) > 2 else None

CAT1_COL, CAT0_COL = "5DADE2", "F5B041"           # rule grids: category 1 (F) / category 0 (J)
CORRECT_COL, WRONG_COL = "2ECC40", "FF4136"
AB_COL = dd.AB_COL

# ---------------------------------------------------------------- task settings from config.js

def read_config():
    js = CONFIG.read_text()

    def block(name):
        m = re.search(rf"export const {name} = (\{{.*?\n\}}|\{{[^\n]*\}});", js, re.S)
        if not m:
            raise ValueError(f"{name} not found in {CONFIG}")
        return m.group(1)

    def num(src, key):
        m = re.search(rf"\b{key}:\s*(-?[\d.]+)", src)
        if not m:
            raise ValueError(f"{key} not found")
        return float(m.group(1))

    rules = {}
    for name, typ, size, labels in re.findall(
            r"(\w+):\s*\{\s*type:\s*'([^']+)',\s*size:\s*(\d+),\s*labels:\s*\[([^\]]+)\]", block("RULES")):
        rules[name] = {"type": typ, "size": int(size), "labels": [int(v) for v in labels.split(",")]}
    seqs = {v: re.findall(r"'(\w+)'", body) for v, body in re.findall(r"(\w):\s*\[([^\]]*)\]", block("SEQUENCES"))}
    blocks = re.search(r"export const BLOCKS = \[(.*?)\n\];", js, re.S).group(1)
    reps = {ph: [int(r) for r in re.findall(rf"phase:\s*'{ph}'[^}}]*?reps:\s*(\d+)", blocks)]
            for ph in ("practice", "test")}
    timing, crit, bonus, keys = block("TIMING"), block("PRACTICE_CRITERION"), block("BONUS"), block("KEYS")
    screen = block("SCREENING")
    tiers = [(float(lo), eval(amount, {})) for lo, amount in
             re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.\s/]+?)\s*\]", re.search(r"tiers:\s*\[(.*?)\n\s*\]", bonus, re.S).group(1))]
    cfg = {
        "version": re.search(r"export const VERSION = '([\d.]+)'", js).group(1),
        "iti": num(timing, "iti") / 1000, "response": num(timing, "response") / 1000,
        "feedback": num(timing, "feedback") / 1000,
        "key1": re.search(r"cat1:\s*'(\w)'", keys).group(1).upper(),
        "key0": re.search(r"cat0:\s*'(\w)'", keys).group(1).upper(),
        "window": int(num(crit, "window")), "min_correct": int(num(crit, "minCorrect")),
        "tiers": tiers, "quiz_tries": int(num(block("COMPREHENSION"), "maxAttempts")),
        "screen_quiz": "quiz: true" in screen, "screen_typeI": "practiceTypeI: true" in screen,
        "screen_timeouts": num(screen, "practiceTimeouts"), "screen_pay": num(screen, "payUsd"),
        "rules": rules, "seqs": seqs,
        "practice_max": [r * 4 for r in reps["practice"]], "test_trials": [r * 16 for r in reps["test"]],
        "assumed_cm": num(block("CALIBRATION"), "assumedDistanceMm") / 10,
    }
    cfg["dist_range_cm"] = [int(v) / 10 for v in re.search(r"plausibleDistanceMm:\s*\[(\d+),\s*(\d+)\]", js).groups()]
    return cfg


CFG = read_config()


def block_bonus(acc):
    """Dollars for one test block: the amount of the highest tier its accuracy reaches"""
    return max([amount for lo, amount in CFG["tiers"] if acc >= lo], default=0.0)


def session_bonus(acc):
    """The whole bonus if every test block has this accuracy"""
    return block_bonus(acc) * len(CFG["test_trials"])


BONUS_MAX = session_bonus(1.0)


def guess_bonus():
    """Average bonus of a pure guesser (50% per round): each block's correct count is binomial"""
    from scipy.stats import binom
    total = 0.0
    for n in CFG["test_trials"]:
        k = np.arange(n + 1)
        total += float(np.sum(binom.pmf(k, n, 0.5) * np.array([block_bonus(x / n) for x in k])))
    return total


# Kernel-mode make-up of every rule (kernel_model/kernel_modes.py, Design('4x4'))
MODES = {"VI": "A + AB", "X": "A + AB", "II": "A + B + AB", "I": "A", "XOR": "AB"}


def modes(rule):
    """Kernel modes of a rule; the B version of a one-sided type swaps A for B (A + AB -> B + AB)."""
    m = MODES[CFG["rules"][rule]["type"]]
    return "B" + m[1:] if rule.endswith("_B") and m.startswith("A") and "B +" not in m else m


# ---------------------------------------------------------------- assets (the task's own fractals)

GROUPS = json.loads((GROUP_DIR / "groups.json").read_text())
QUADS, PAIRS = GROUPS["groups"]["4"], GROUPS["groups"]["2"]
CAND = GROUPS["candidates"]                          # task file number -> candidate in the pool
FRACT_A = [f"vf{k}" for k in QUADS[0]]
FRACT_B = [f"vf{k}" for k in QUADS[1]]
PRACT = [f"vf{k}" for k in PAIRS[0] + PAIRS[1]]


def crop_square(src, dst, px=360):
    im = Image.open(src).convert("RGBA")
    a = np.array(im)[..., 3]
    r, c = np.where(a.any(1))[0], np.where(a.any(0))[0]
    im = im.crop((c[0], r[0], c[-1] + 1, r[-1] + 1))
    side = max(im.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    canvas.resize((px, px), Image.LANCZOS).save(dst)


def make_assets():
    """Cropped copies of the task's fractals (vf<k>), the same shapes in the earlier colours
    (of<k>, from the original pool: same seed, same shapes), the distance histogram, the contact
    sheet and the similarity heatmaps, in the slide-asset folder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dd.ASSETS.mkdir(parents=True, exist_ok=True)
    used = {int(n[2:]) for n in FRACT_A + FRACT_B + PRACT}
    for k in used:
        crop_square(GROUP_DIR / f"{k}.png", dd.ASSETS / f"vf{k}.png")
        crop_square(POOL_OLD / "candidates" / f"{CAND[str(k)]}.png", dd.ASSETS / f"of{k}.png")
    # distance histogram: all candidate pairs, within the task's groups, between them
    D = np.load(POOL_VIVID / "dreamsim_distances.npy")
    iu = np.triu_indices(len(D), 1)
    files = {int(f): c - 1 for f, c in CAND.items()}
    groups = [[files[f] for f in g] for g in QUADS + PAIRS]
    within = [D[a, b] for g in groups for i, a in enumerate(g) for b in g[i + 1:]]
    member = {m: gi for gi, g in enumerate(groups) for m in g}
    sel = sorted(member)
    between = [D[a, b] for i, a in enumerate(sel) for b in sel[i + 1:] if member[a] != member[b]]
    plt.rcParams.update({"font.size": 11, "text.color": "white", "axes.labelcolor": "white",
                         "xtick.color": "white", "ytick.color": "white", "axes.edgecolor": "#BFBFBF"})
    fig, ax = plt.subplots(figsize=(5.4, 2.4), facecolor="black")
    ax.set_facecolor("black")
    ax.hist(D[iu], bins=80, density=True, color="#7F7F7F", label="all candidate pairs")
    ax.hist(between, bins=50, density=True, color="#FFAB40", alpha=0.6, label="between groups")
    ax.axvspan(min(within), max(within), color="#3CC9A0", alpha=0.9, label="within a group")
    ax.set_xlabel("DreamSim distance")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(dd.ASSETS / "vivid_distance_hist.png", dpi=200, facecolor="black")
    plt.close(fig)
    Image.open(GROUP_DIR / "contact_sheet.png").save(dd.ASSETS / "vivid_contact_sheet.png")
    # similarity heatmaps, each sorted by its own clustering (fractal_stimuli/similarity_matrices.py)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="black")
    for ax, name in zip(axes, ("colour", "shape", "perceptual")):
        s = np.load(POOL_VIVID / "similarity" / f"similarity_{name}.npy").astype(float)
        d = np.clip(1 - s, 0, None)
        np.fill_diagonal(d, 0)
        order = leaves_list(linkage(squareform(d, checks=False), method="average"))
        off = s[~np.eye(len(s), dtype=bool)]
        ax.imshow(s[np.ix_(order, order)], cmap="magma", vmin=np.percentile(off, 1), vmax=np.percentile(off, 99),
                  interpolation="nearest")
        ax.set_title(name, color="white", fontsize=13)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(dd.ASSETS / "vivid_similarity.png", dpi=170, facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------- drawing helpers

def new_slide(prs, ttl=None):
    """Slide on the template's TITLE_ONLY (or BLANK) layout, looked up by name."""
    layouts = {l.name: l for l in prs.slide_layouts}
    s = prs.slides.add_slide(layouts["TITLE_ONLY"] if ttl else layouts["BLANK"])
    for ph in list(s.placeholders):
        if ph.placeholder_format.type != 1:
            ph._element.getparent().remove(ph._element)
    if ttl:
        dd.title(s, ttl)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = dd.rgb("000000")
    return s


def oval(slide, cx, cy, d, fill=None, line_col=None, line_w=1.0):
    return shape(slide, MSO_SHAPE.OVAL, cx - d / 2, cy - d / 2, d, d, fill=fill, line=line_col, line_w=line_w)


def bullseye(slide, cx, cy, u):
    """MATLAB-task fixation: 0.6 deg black disc, white cross 0.15 deg thick, 0.2 deg black centre."""
    d, bar = max(0.6 * u, 0.08), max(0.15 * u, 0.02)
    oval(slide, cx, cy, d, fill="000000")
    shape(slide, MSO_SHAPE.RECTANGLE, cx - d / 2, cy - bar / 2, d, bar, fill=WHITE)
    shape(slide, MSO_SHAPE.RECTANGLE, cx - bar / 2, cy - d / 2, bar, d, fill=WHITE)
    oval(slide, cx, cy, d / 3, fill="000000")


def display(slide, cx, cy, u, files=None, fix=False, outcome=None, word=None, labels=False):
    """The task screen, u inches per degree: grey display (24 x 12 deg), symbols 4 deg at +-6 deg,
    optional bull's eye, feedback circles and word. Returns the display's box."""
    sw, sh = 24 * u, 12 * u
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, cx - sw / 2, cy - sh / 2, sw, sh, fill=SCREEN, adj=0.04)
    if files:
        for f, x in zip(files, (cx - 6 * u, cx + 6 * u)):
            image(slide, f, x, cy, 4 * u)
    if outcome:
        for x in (cx - 6 * u, cx + 6 * u):
            oval(slide, x, cy, 5.4 * u, line_col=outcome, line_w=max(1.0, 0.3 * u * 72))
    if word:
        text(slide, cx - 3 * u, cy - 0.8 * u, 6 * u, 1.6 * u, word, size=max(7, int(u * 30)), color=WHITE,
             bold=True, align="ctr", anchor="ctr")
    if fix:
        bullseye(slide, cx, cy, u)
    if labels:
        text(slide, cx - 6 * u - 1, cy - 2.9 * u - 0.35, 2, 0.3, "A", size=16, color=A_COL, bold=True, align="ctr")
        text(slide, cx + 6 * u - 1, cy - 2.9 * u - 0.35, 2, 0.3, "B", size=16, color=B_COL, bold=True, align="ctr")
    return cx - sw / 2, cy - sh / 2, sw, sh


def rule_grid(slide, x, y, cell, labels, size, gap=0.02):
    """Category grid: rows = left (A) symbol, columns = right (B) symbol; blue = F, orange = J."""
    for a in range(size):
        for b in range(size):
            shape(slide, MSO_SHAPE.RECTANGLE, x + b * cell, y + a * cell, cell - gap, cell - gap,
                  fill=CAT1_COL if labels[size * a + b] == 1 else CAT0_COL)


def key_cap(slide, x, y, letter, size=0.55, label=None, label_col=WHITE):
    k = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, size, size, fill="1E1E1E", line=WHITE, line_w=1.5, adj=0.15)
    label_in(k, letter, size=int(size * 36), color=WHITE, bold=True)
    if label:
        text(slide, x - 0.5, y + size + 0.05, size + 1.0, 0.3, label, size=13, color=label_col, align="ctr", bold=True)


def pill(slide, x, y, w, h, txt, fill, size=14, color="111111"):
    p = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, adj=0.5)
    label_in(p, txt, size=size, color=color, bold=True)
    return p


def arrow(slide, x1, y, x2):
    line(slide, x1, y, x2, y, color=GREY, width=1.5, arrow=True)


def legend_f_j(slide, x, y):
    shape(slide, MSO_SHAPE.RECTANGLE, x, y, 0.3, 0.3, fill=CAT1_COL)
    text(slide, x + 0.4, y - 0.04, 2.2, 0.4, f"category 1 → {CFG['key1']}", size=14, anchor="ctr")
    shape(slide, MSO_SHAPE.RECTANGLE, x, y + 0.45, 0.3, 0.3, fill=CAT0_COL)
    text(slide, x + 0.4, y + 0.41, 2.2, 0.4, f"category 0 → {CFG['key0']}", size=14, anchor="ctr")


def lstar_grey(L):
    """sRGB hex of the neutral grey with CIE lightness L*."""
    y = ((L + 16) / 116) ** 3 if L > 8 else L / 903.3
    v = 12.92 * y if y <= 0.0031308 else 1.055 * y ** (1 / 2.4) - 0.055
    g = round(255 * min(1, max(0, v)))
    return f"{g:02X}" * 3


def fractal_row(slide, names, x, y, size, gap):
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x - 0.15, y - 0.15, len(names) * (size + gap) - gap + 0.3,
          size + 0.3, fill=SCREEN, adj=0.08)
    for i, f in enumerate(names):
        image(slide, f, x + i * (size + gap) + size / 2, y + size / 2, size)


def picture(slide, name, cx, top, width=None, height=None):
    kw = {"width": dd.Inches(width)} if width else {"height": dd.Inches(height)}
    pic = slide.shapes.add_picture(str(dd.ASSETS / name), 0, dd.Inches(top), **kw)
    pic.left = dd.Inches(cx) - pic.width // 2
    return pic


# ---------------------------------------------------------------- slides: the task

def s_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    for ph in list(s.placeholders):
        if ph.placeholder_format.type not in (3, 4):
            ph._element.getparent().remove(ph._element)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = dd.rgb("000000")
    for ph in s.placeholders:
        ttl = ph.placeholder_format.type == 3
        ph.text = "Category learning and generalization" if ttl else "A 4×4 online study (study 1)"
        for p in ph.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.name = dd.FONT
                r.font.size = Pt(34 if ttl else 18)
                r.font.color.rgb = dd.rgb(WHITE)
    for i, f in enumerate(FRACT_A + FRACT_B):
        image(s, f, 1.65 + i * 0.95, 4.55, 0.66)


def s_screen(prs):
    s = new_slide(prs, "The screen")
    u = 0.26
    cx, cy = 5.0, 3.2
    display(s, cx, cy, u, files=[FRACT_A[0], FRACT_B[0]], labels=True)
    y = cy + 2.55 * u + 0.18
    line(s, cx, y, cx - 6 * u, y, color=WHITE, width=1.25, arrow=True)
    text(s, cx - 6 * u, y + 0.02, 6 * u, 0.28, "6°", size=13, align="ctr")
    text(s, cx + 6 * u - 1, cy + 2 * u + 0.02, 2, 0.28, "4°", size=13, align="ctr")
    text(s, 0.6, 4.95, 8.8, 0.4, "Two symbols, left (A) and right (B) of the centre · sizes in degrees of visual angle",
         size=14, color=GREY, align="ctr")


def s_task(prs):
    s = new_slide(prs, f"Every pair is category 1 ({CFG['key1']}) or category 0 ({CFG['key0']})")
    cell, x0, y0 = 0.58, 3.3, 1.95
    for i, f in enumerate(FRACT_A):
        image(s, f, x0 - 0.45, y0 + (i + 0.5) * cell, 0.48)
    for j, f in enumerate(FRACT_B):
        image(s, f, x0 + (j + 0.5) * cell, y0 - 0.38, 0.48)
    rule_grid(s, x0, y0, cell, CFG["rules"]["x4_X_A"]["labels"], 4, gap=0.04)
    text(s, x0 - 1.25, y0 + 2 * cell - 0.2, 0.5, 0.4, "A", size=20, color=A_COL, bold=True, align="r")
    text(s, x0 - 0.75, y0 - 0.6, 0.6, 0.4, "B", size=20, color=B_COL, bold=True, align="r")
    key_cap(s, 6.55, 2.0, CFG["key1"], size=0.7, label="category 1", label_col=CAT1_COL)
    key_cap(s, 7.75, 2.0, CFG["key0"], size=0.7, label="category 0", label_col=CAT0_COL)
    text(s, 6.0, 3.25, 3.4, 0.8, [("the same keys for everyone\n", {"size": 14}),
                                  ("learned from feedback after every answer", {"size": 12, "color": GREY})],
         align="ctr")
    text(s, 0.6, 4.75, 8.8, 0.4, "4 symbols per side → 16 pairs · a rule assigns each pair a category · "
         "fractals go to rows and columns at random", size=13, color=GREY, align="ctr")


def s_round(prs):
    s = new_slide(prs, "One round")
    u = 0.105
    steps = [
        ("Fixation", f"{CFG['iti']:g} s · look at the centre", dict(fix=True)),
        ("The pair: free viewing", f"{CFG['key1']} or {CFG['key0']}, ≤ {CFG['response']:g} s",
         dict(files=[FRACT_A[0], FRACT_B[0]])),
        ("Feedback", f"{CFG['feedback']:g} s", dict(files=[FRACT_A[0], FRACT_B[0]], outcome=CORRECT_COL)),
    ]
    for i, (name, dur, kw) in enumerate(steps):
        cx, cy = 2.0 + i * 1.9, 2.0 + i * 1.05
        display(s, cx, cy, u, **kw)
        text(s, cx + 1.4, cy - 0.5, 3.2, 0.7, [(name + "\n", {"size": 15}), (dur, {"size": 13, "color": GREY})])
    line(s, 0.55, 2.3, 4.6, 5.0, color=WHITE, width=1.5, arrow=True)
    text(s, 5.6, 4.95, 4.0, 0.35, "rounds follow each other automatically", size=13, color=GREY, align="r")


def s_feedback(prs):
    s = new_slide(prs, "Feedback after every answer")
    u = 0.17
    items = [(CORRECT_COL, "right answer"), (WRONG_COL, f"wrong, or no answer within {CFG['response']:g} s")]
    for i, (col, sub) in enumerate(items):
        cx = 2.6 + i * 4.8
        display(s, cx, 2.75, u, files=[FRACT_A[1], FRACT_B[2]], outcome=col)
        text(s, cx - 2.0, 3.9, 4.0, 0.35, sub, size=14, color=col, align="ctr")
    text(s, 0.6, 4.5, 8.8, 0.7, [("only green or red circles around both symbols: no words, no points\n",
                                  {"size": 14}),
                                 ("a break screen between blocks shows that block's accuracy (and, in the test, "
                                  "its bonus)", {"size": 12, "color": GREY})], align="ctr")


def s_bonus(prs):
    s = new_slide(prs, "Bonus: per test block, by accuracy")
    n_blocks = len(CFG["test_trials"])
    bounds = [lo for lo, _ in CFG["tiers"]]
    rows = [(f"below {bounds[0]:.0%}", 0.0)]
    for i, (lo, amount) in enumerate(CFG["tiers"]):
        hi = bounds[i + 1] if i + 1 < len(bounds) else None
        rows.append((f"{lo:.0%}–{hi:.0%}" if hi else f"{lo:.0%} or more", amount))
    text(s, 2.9, 1.35, 2.4, 0.4, "block accuracy", size=13, color=GREY, align="r")
    text(s, 5.6, 1.35, 1.8, 0.4, "bonus", size=13, color=GREY)
    for i, (lab, amount) in enumerate(rows):
        y = 1.8 + i * 0.5
        text(s, 2.9, y, 2.4, 0.45, lab, size=17, anchor="ctr", align="r")
        text(s, 5.6, y, 1.6, 0.45, f"${amount:.2f}", size=17, color=CORRECT_COL if amount else GREY,
             anchor="ctr", bold=True)
    text(s, 0.6, 3.95, 8.8, 1.2,
         [(f"up to ${BONUS_MAX:.0f} over the {n_blocks} test blocks · no penalties · late answers count as wrong\n",
           {"size": 14}),
          (f"a pure guesser averages ~${guess_bonus():.2f} · told up front · each block's bonus on its break screen · "
           "the total at the end · paid through Prolific on top of the base pay", {"size": 12, "color": GREY})],
         align="ctr")


def s_screening(prs):
    s = new_slide(prs, "Early screening")
    items = []
    if CFG["screen_quiz"]:
        items.append(("Quiz failed twice", f"all questions right within {CFG['quiz_tries']} tries, right after the "
                                           "instructions"))
    if CFG["screen_typeI"]:
        items.append(("Type I practice not passed", f"{CFG['min_correct']} of the last {CFG['window']} right within "
                                                    f"{CFG['practice_max'][0]} rounds"))
    items.append((f"More than {CFG['screen_timeouts']:.0%} of practice rounds too slow", "no answer within "
                                                                                       f"{CFG['response']:g} s"))
    for i, (head, sub) in enumerate(items):
        y = 1.4 + i * 0.8
        shape(s, MSO_SHAPE.OVAL, 1.0, y + 0.1, 0.22, 0.22, fill=WRONG_COL)
        text(s, 1.4, y, 7.8, 0.75, [(head + "\n", {"size": 17}), (sub, {"size": 12, "color": GREY})])
    text(s, 0.6, 3.95, 8.8, 1.2,
         [(f"screened out: Prolific custom screening, ${CFG['screen_pay']:.2f} fixed, doesn't use up a place\n",
           {"size": 14}),
          ("the XOR practice is not a screen · without screening, a colleague's study lost about half its "
           "participants", {"size": 12, "color": GREY})], align="ctr")


def s_practice(prs):
    s = new_slide(prs, "Practice: two 2×2 blocks")
    r = CFG["rules"]
    items = [("Type I", "A or B, at random", r["x2_I_A"]["labels"], PRACT[:2], PRACT[2:4]),
             ("XOR", "both sides matter", r["x2_XOR"]["labels"], PRACT[2:4], PRACT[:2])]
    for i, (name, sub, labels, left, right) in enumerate(items):
        x = 2.0 + i * 3.6
        pill(s, x, 1.45, 2.2, 0.5, name, A_COL if i == 0 else AB_COL, size=17)
        rule_grid(s, x + 0.6, 2.25, 0.5, labels, 2, gap=0.04)
        text(s, x - 0.2, 3.35, 2.6, 0.35, sub, size=13, color=GREY, align="ctr")
        if i == 0:
            arrow(s, x + 2.35, 1.7, x + 3.45)
    legend_f_j(s, 0.45, 2.3)
    text(s, 0.6, 4.05, 8.8, 0.9,
         [(f"Each ends once {CFG['min_correct']} of the last {CFG['window']} answers are right\n", {"size": 17}),
          (f"at most {CFG['practice_max'][0]} rounds · fractal pairs · no bonus · type I must be passed "
           "(screening)", {"size": 13, "color": GREY})],
         align="ctr")


def s_sequence(prs):
    s = new_slide(prs, "Test: three 4×4 blocks, VI → X → II")
    r = CFG["rules"]
    for b in range(3):
        text(s, 2.25 + b * 2.35, 1.3, 1.8, 0.3, f"block {b + 1}", size=13, color=GREY, align="ctr")
    for row, (ver, seq) in enumerate(CFG["seqs"].items()):
        y = 1.7 + row * 1.45
        text(s, 0.3, y + 0.35, 1.7, 0.5, f"version {ver}", size=15, color=A_COL if ver == "A" else B_COL,
             anchor="ctr", align="r", bold=True)
        for b, rule in enumerate(seq):
            x = 2.6 + b * 2.35
            rule_grid(s, x, y, 0.26, r[rule]["labels"], 4)
            text(s, x - 0.4, y + 1.08, 1.85, 0.3, f"{r[rule]['type']} · {modes(rule)}", size=12, align="ctr")
            if b < 2:
                arrow(s, x + 1.2, y + 0.52, x + 2.2)
    legend_f_j(s, 0.4, 4.45)
    text(s, 3.2, 4.45, 6.4, 0.8,
         [(f"{CFG['test_trials'][0]} rounds per block · new fractals every block\n", {"size": 13}),
          ("type II weighs A and B equally: block 3 is the same for everyone · one Prolific study per version "
           "(?version=A / B)", {"size": 11, "color": GREY})])


def s_fractals_why(prs):
    s = new_slide(prs, "Fractals: matched, and easy to see")
    rows = [("before", [f"of{n[2:]}" for n in FRACT_A], "L* 62 / 48 / 68 · chroma 40\nclose to the grey: dull"),
            ("now", FRACT_A, "L* 70 / 56 / 78 · chroma 60\nbrighter than the grey, vivid")]
    for i, (lab, names, stats) in enumerate(rows):
        y = 1.55 + i * 1.45
        text(s, 0.3, y + 0.3, 1.3, 0.4, lab, size=16, color=GREY if i == 0 else WHITE, align="r")
        fractal_row(s, names, 1.95, y, 0.85, 0.25)
        text(s, 6.55, y + 0.15, 3.3, 0.7, stats, size=12, color=GREY if i == 0 else WHITE)
    text(s, 0.6, 4.6, 8.8, 0.6, [("the same shapes, recoloured: every fractal shares one lightness profile, "
                                  "so none is brighter than another\n", {"size": 12}),
                                 ("background grey L* 53.6", {"size": 11, "color": GREY})], align="ctr")


def s_fractals_how(prs):
    s = new_slide(prs, "How each fractal is made")
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, 1.45, 3.0, 3.0, fill=SCREEN, adj=0.06)
    image(s, FRACT_A[1], 2.2, 2.95, 2.4)
    text(s, 0.7, 4.55, 3.0, 0.3, "1,500 candidates", size=12, color=GREY, align="ctr")
    x = 4.3
    text(s, x, 1.4, 5.3, 0.6, [("3 nested polygons with bent edges\n", {"size": 16}),
                               ("after Miyashita et al. (1997)", {"size": 12, "color": GREY})])
    text(s, x, 2.15, 5.3, 0.35, "same lightness in every fractal (CIE L*)", size=15)
    for i, (lab, L) in enumerate((("outer", 70), ("middle", 56), ("inner", 78), ("background", 53.6))):
        bx = x + i * 1.3
        sw = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, bx, 2.6, 1.1, 0.5, fill=lstar_grey(L), adj=0.15,
                   line=WHITE if lab == "background" else None, line_w=1)
        label_in(sw, f"{L:g}", size=13, color="111111", bold=True)
        text(s, bx - 0.1, 3.12, 1.3, 0.3, lab, size=11, color=GREY, align="ctr")
    text(s, x, 3.6, 5.3, 0.9, [("3 of 12 evenly spaced hues · chroma 60\n", {"size": 15}),
                               ("lowered to what the screen can show where needed (33–60, cyan–blue lowest)\n",
                                {"size": 11, "color": GREY}),
                               ("same filled area: 22% of the image", {"size": 15})])


def s_fractals_distinct(prs):
    s = new_slide(prs, "Equally distinct groups")
    fractal_row(s, FRACT_A, 0.85, 1.55, 0.72, 0.2)
    text(s, 0.6, 2.5, 3.9, 0.7, [("every pair in a group: 0.257 ± 0.0075\n", {"size": 15}),
                                 ("so every block is equally hard", {"size": 12, "color": GREY})], align="ctr")
    text(s, 0.6, 3.35, 3.9, 1.2, [("distance = DreamSim\n", {"size": 14}),
                                  ("DINO + CLIP + OpenCLIP, tuned on human similarity judgements; "
                                   "1 − cosine, on the task's grey background", {"size": 11, "color": GREY})],
         align="ctr")
    picture(s, "vivid_distance_hist.png", 7.2, 1.5, width=4.6)
    text(s, 4.75, 3.65, 4.9, 0.8, [("no two of the 96 chosen closer than 0.161: no near-duplicates\n",
                                    {"size": 12}),
                                   ("all candidate pairs: median 0.257, 5–95% range 0.161–0.364",
                                    {"size": 11, "color": GREY})], align="ctr")


def s_fractals_similarity(prs):
    s = new_slide(prs, "What the distance tracks")
    picture(s, "vivid_similarity.png", 5.0, 1.3, width=8.6)
    text(s, 0.6, 4.35, 8.8, 0.9,
         [("all 1,500 candidates, sorted by clustering · DreamSim correlates 0.46 with colour, 0.03 with shape\n",
           {"size": 13}),
          ("perceptual clusters are colour schemes: equal distances mostly mean equally different colours",
           {"size": 12, "color": GREY})], align="ctr")


def s_fractals_set(prs):
    s = new_slide(prs, "The final set")
    pic = picture(s, "vivid_contact_sheet.png", 5.0, 1.2, width=9.2)
    if pic.height > dd.Inches(3.55):                 # tall sheet: fit the height instead
        pic._element.getparent().remove(pic._element)
        picture(s, "vivid_contact_sheet.png", 5.0, 1.2, height=3.55)
    text(s, 0.6, 4.85, 8.8, 0.4, "18 groups of 4 (test) · 12 pairs (practice) · each side of a block shows one "
         "whole group, never reused for that participant", size=12, color=GREY, align="ctr")


def s_dva(prs):
    s = new_slide(prs, "Sizes in degrees of visual angle")
    text(s, 0.6, 1.3, 4.2, 0.35, "1 · match a real card", size=16, align="ctr")
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 1.55, 1.85, 2.2, 1.39, fill="DDDDDD", adj=0.06)
    shape(s, MSO_SHAPE.RECTANGLE, 3.5, 3.0, 0.25, 0.24, line="FF0000", line_w=2.5)
    text(s, 0.6, 3.35, 4.2, 0.35, "→ pixels per mm", size=14, color=GREY, align="ctr")
    text(s, 5.2, 1.3, 4.2, 0.35, "2 · find the blind spot", size=16, align="ctr")
    shape(s, MSO_SHAPE.RECTANGLE, 8.35, 2.35, 0.35, 0.35, fill="000000", line=WHITE, line_w=2)
    oval(s, 6.6, 2.52, 0.33, fill="FF2222", line_col=WHITE, line_w=1.5)
    line(s, 6.35, 2.95, 5.75, 2.95, color=GREY, width=1.5, arrow=True)
    text(s, 5.2, 3.05, 4.2, 0.6, [("cover the right eye, look AT the square,\n", {"size": 12}),
                                  ("SPACE when the ball vanishes → viewing distance", {"size": 12})],
         color=GREY, align="ctr")
    text(s, 0.6, 3.95, 8.8, 0.45, "pixels per degree = 2 · distance · tan(0.5°) · pixels per mm", size=16, align="ctr")
    lo, hi = CFG["dist_range_cm"]
    text(s, 0.6, 4.45, 8.8, 0.7,
         [(f"Outside {lo:g}–{hi:g} cm: measure again once, then assume {CFG['assumed_cm']:g} cm (flagged)\n", {}),
          ("too big for the window: everything shrinks by one factor (layout_scale, saved)", {})],
         size=12, color=GREY, align="ctr")


def s_quiz(prs):
    s = new_slide(prs, "Instructions, then a quiz")
    qs = [("What do you do when the two symbols appear?", f"press {CFG['key1']} or {CFG['key0']}, by the pair"),
          ("How do you know whether you were right?", "green or red circles around the symbols"),
          ("How is the bonus earned?", "by each test block's accuracy"),
          ("Where do you look between rounds?", "at the target in the centre")]
    for i, (q, a) in enumerate(qs):
        y = 1.4 + i * 0.75
        shape(s, MSO_SHAPE.OVAL, 1.0, y + 0.1, 0.22, 0.22, fill=A_COL)
        text(s, 1.4, y, 7.8, 0.7, [(q + "\n", {"size": 16}), (a, {"size": 12, "color": GREY})])
    text(s, 0.6, 4.6, 8.8, 0.4, f"all four right to continue · {CFG['quiz_tries']} tries, then screened out "
         f"(${CFG['screen_pay']:.2f})", size=12, color=GREY, align="ctr")


def s_online(prs):
    s = new_slide(prs, "Built for online data")
    items = [
        ("Desktop only", "browser check, window ≥ 1000 × 650 px"),
        ("Full screen", "exits are counted on every trial"),
        ("Tab switches", "counted on every trial"),
        ("Every trial saved", "pair, key, RT, px / degree"),
        ("DataPipe → Google Drive", "a participant who quits still leaves their trials"),
        ("Reproducible", "seed rebuilds the design · Prolific IDs on every row"),
    ]
    for i, (head, sub) in enumerate(items):
        col, row = i % 2, i // 2
        x, y = 0.9 + col * 4.4, 1.5 + row * 1.1
        shape(s, MSO_SHAPE.OVAL, x, y + 0.08, 0.22, 0.22, fill=A_COL)
        text(s, x + 0.4, y - 0.05, 3.9, 0.9, [(head + "\n", {"size": 16}), (sub, {"size": 12, "color": GREY})])


def s_session(prs):
    s = new_slide(prs, "The session")
    steps = [("Consent", None), ("Screen setup", "~2 min"), ("Instructions + quiz", "screening"),
             ("Practice", "2×2 · screening"), ("Test", f"3 × {CFG['test_trials'][0]}"), ("Bonus → Prolific", None)]
    x = 0.45
    widths = [1.15, 1.45, 1.75, 1.15, 1.15, 1.6]
    for (name, sub), w in zip(steps, widths):
        pill(s, x, 2.15, w, 0.6, name, "3A3A3A", size=12, color=WHITE)
        if sub:
            text(s, x, 2.85, w, 0.3, sub, size=11, color=GREY, align="ctr")
        x += w + 0.13
    text(s, 0.6, 3.9, 8.8, 0.5, f"about {MINUTES} minutes", size=20, align="ctr")
    text(s, 0.6, 4.45, 8.8, 0.35, "rounds run automatically · rest between blocks", size=13, color=GREY, align="ctr")


# ---------------------------------------------------------------- participants and cost
# The Prolific set-up (from the study pages): two studies, one per version, 10 people each,
# $15.00 for ~40 min ($22.50/h), academic fee 33.3% on everything paid, bonuses included.
MINUTES = 56                            # expected session (catlearn_4x4_prolific/README.md: 3 x 208 test rounds, practice, setup)
LISTED_MINUTES = 60                     # the time listed on Prolific
N_PER_STUDY = 10
PILOT = ("A", 5, 5)                     # the first run: study A only, 5 people, 5 screen-out slots
PAY = 20.00
FEE = 1 / 3


SCREENED_SHARE = 0.5                    # if half the starters were screened out (the colleague's loss)


def s_participants(prs):
    s = new_slide(prs, "Participants and cost")
    n = N_PER_STUDY * len(CFG["seqs"])
    for i, ver in enumerate(CFG["seqs"]):
        x = 1.0 + i * 4.4
        text(s, x, 1.3, 3.6, 0.4, f"Prolific study {ver}", size=17, color=A_COL if ver == "A" else B_COL, align="ctr")
        text(s, x, 1.7, 3.6, 0.7, f"{N_PER_STUDY}", size=40, bold=True, align="ctr", anchor="ctr")
        text(s, x, 2.45, 3.6, 0.35, f"${N_PER_STUDY * PAY:,.0f} + ${N_PER_STUDY * PAY * FEE:,.0f} fee "
             f"= ${N_PER_STUDY * PAY * (1 + FEE):,.0f}", size=14, align="ctr")
    base = n * PAY * (1 + FEE)
    low, mx = n * guess_bonus() * (1 + FEE), n * BONUS_MAX * (1 + FEE)
    screened = n * SCREENED_SHARE / (1 - SCREENED_SHARE) * CFG["screen_pay"] * (1 + FEE)
    top = CFG["tiers"][-1][0]
    text(s, 0.6, 3.0, 8.8, 1.4,
         [(f"{n} people · ${PAY:.2f} each for {LISTED_MINUTES} min (${PAY * 60 / LISTED_MINUTES:.2f}/h) · base "
           f"${base:,.0f} with fees\n", {"size": 15}),
          (f"bonus: ~${low:,.0f} if everyone guessed, ${mx:,.0f} if everyone reaches {top:.0%} in every block\n",
           {"size": 14}),
          (f"screened out at ${CFG['screen_pay']:.2f} each: ~${screened:,.0f} if half of all starters fail a screen\n",
           {"size": 14}),
          (f"total ${base + low + screened:,.0f}–{base + mx + screened:,.0f}", {"size": 16, "bold": True})],
         align="ctr")
    pv, pn, ps = PILOT
    text(s, 0.6, 4.45, 8.8, 0.7,
         [(f"first run: study {pv} only, {pn} people + {ps} screen-out slots · "
           f"${(pn * PAY + ps * CFG['screen_pay']) * (1 + FEE):,.0f} with fees · bonus up to "
           f"${pn * BONUS_MAX * (1 + FEE):,.0f}\n", {"size": 13, "color": A_COL}),
          ("fees included · the bonus range runs from everyone guessing to everyone in the top tier",
           {"size": 11, "color": GREY})], align="ctr")


def s_prolific(prs):
    s = new_slide(prs, "Prolific set-up")
    rows = [("Study", "\"Category learning and generalization\" · internal catlearn_4x4_v1.0_A / _B"),
            ("Places", f"first run: {PILOT[1]} in study {PILOT[0]} + {PILOT[2]} screen-out slots · then {N_PER_STUDY} "
                       f"per study · {LISTED_MINUTES} min listed · ${PAY:.2f}"),
            ("Settings", "desktop only · label \"Decision making\" · IDs in the URL · manual review"),
            ("Link", "…/catlearn_4x4_prolific/?PROLIFIC_PID=…&STUDY_ID=…&SESSION_ID=…&version=A (or B)"),
            ("Codes", "completion: study A CCP8AO6I, study B its own · screen-out: one per study"),
            ("Screening", f"custom screening path, ${CFG['screen_pay']:.2f} fixed · doesn't use up a place"),
            ("Filters", "fluent English · approval 95–100% · no colour-vision issues · normal or corrected vision · "
                        "UK + US · standard sample")]
    for i, (head, body) in enumerate(rows):
        y = 1.25 + i * 0.55
        text(s, 0.6, y, 1.6, 0.5, head, size=15, color=A_COL, bold=True, align="r")
        text(s, 2.4, y + 0.03, 7.3, 0.6, body, size=12 if head == "Link" else 13)


def s_next(prs):
    s = new_slide(prs, "Next: where people look")
    text(s, 0.6, 1.6, 8.8, 1.2, [("study 2: webcam gaze tracking\n", {"size": 22}),
                                 ("WebGazer through jsPsych's webgazer extension", {"size": 15, "color": GREY})],
         align="ctr")
    text(s, 0.6, 3.0, 8.8, 1.2,
         [("which symbol people look at as they learn: A, B, or both\n", {"size": 15}),
          ("left out of study 1 to keep the session simple and calibration-free", {"size": 12, "color": GREY})],
         align="ctr")


SLIDES = [s_title, s_screen, s_task, s_round, s_feedback, s_bonus, s_practice, s_screening, s_sequence,
          s_fractals_why, s_fractals_how, s_fractals_distinct, s_fractals_similarity, s_fractals_set,
          s_dva, s_quiz, s_online, s_session, s_participants, s_prolific, s_next]


def main():
    make_assets()
    prs = Presentation(str(TEMPLATE))
    ids = prs.slides._sldIdLst
    for sld in list(ids):
        prs.part.drop_rel(sld.get(qn("r:id")))
        ids.remove(sld)
    for i, make in enumerate(SLIDES, 1):
        if ONLY is None or str(i) in ONLY.split(","):
            make(prs)
    prs.save(str(OUT))
    print(f"saved {OUT} ({len(prs.slides)} slides; task version {CFG['version']})")


if __name__ == "__main__":
    main()
