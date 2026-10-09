"""Per-participant details: session, setup, time per block, practice, attention, bonus,
performance and (if a Prolific export is there) demographics.

From each finished session's CSV (who was screened out or left, and why, is in B06):
  session      task version, start (UTC), browser, OS, screen, refresh rate
  setup        viewing distance (virtual chinrest), calibration attempts, deg per px,
               comprehension-quiz attempts; minutes from the start to the first block
  time         minutes in each block (end of its intro screen -> last trial) and on the
               break before it (last trial of the previous block -> end of this block's
               intro: summary, screen check, intro), and in total
  practice     trials to criterion and accuracy, per practice block
  test         accuracy, late answers and bonus per test block; P(F) and the exclusion
               (helpers.data.exclusions); whether accuracy over all test trials is above
               chance (one-sided binomial, p < .05)
  attention    window/tab switches and fullscreen exits
  bonus        what the task promised (its final bonus_usd), per block and in total
Demographics: Prolific's export ("Download demographic data" on the study's Submissions
page), saved as any .csv in params.PROLIFIC_DIR; merged on the participant ID. Without it the
demographic columns are left out.

Outputs, in analysis/average_version_<A|B>/sanity_tests/:
  participants.csv   every value, one row per participant
  participants.md    the same as readable tables
  timing.png         minutes per part of the session (box plots over participants; red:
                     excluded)

  ~/miniforge3/envs/kernelbehav/bin/python B05_participants.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from helpers.data import load_file, session_files
from helpers.plots import apply_dark_theme, participant_lines, save
from helpers.report import md_table
from helpers.sessions import demographics, performance, session_info
from helpers.study import Study, parser
from params import BOX_COLOR, PROLIFIC_DIR

DEMOGRAPHICS = ("age", "sex", "ethnicity_simplified", "country_of_residence", "nationality", "language",
                "fluent_languages", "student_status", "employment_status", "prolific_status", "prolific_time_taken_s")
SECTIONS = {
    "Session": ["id", "task_version", "started_at", "browser", "os", "screen", "refresh_hz"],
    "Setup and attention": ["id", "view_dist_cm", "calibration_attempts", "px_per_deg", "quiz_attempts",
                            "blur_count", "fullscreen_exits"],
    "Time (minutes)": ["id", "setup_min", "practice1_min", "practice2_min", "test1_break_min", "test1_min",
                       "test2_break_min", "test2_min", "test3_break_min", "test3_min", "total_min"],
    "Practice": ["id", "practice1_rule", "practice1_trials", "practice1_acc", "practice2_rule", "practice2_trials",
                 "practice2_acc"],
    "Test and bonus": ["id", "test1_acc", "test2_acc", "test3_acc", "test_acc", "above_chance", "p_f", "excluded",
                       "test1_bonus_usd", "test2_bonus_usd", "test3_bonus_usd", "bonus_usd"],
}


def write_md(df, path, title):
    df = df.assign(id=df["participant"].str[:8])
    sections = dict(SECTIONS)
    demo = [c for c in DEMOGRAPHICS if c in df]
    if demo:
        sections["Demographics (Prolific)"] = ["id"] + demo
    parts = [f"# {title}", ""]
    for name, cols in sections.items():
        parts += [f"## {name}", "", md_table(df[[c for c in cols if c in df]]), ""]
    if not demo:
        parts += ["Demographics: no Prolific export found in " + str(PROLIFIC_DIR) + ".", ""]
    parts += [f"Bonus promised in total: ${df['bonus_usd'].sum():.2f} ({len(df)} participants).", ""]
    path.write_text("\n".join(parts))


def plot_timing(df, title, path):
    parts = {"setup_min": "Setup", "practice1_min": "Practice 1", "practice2_min": "Practice 2",
             "test1_min": "Test 1", "test2_min": "Test 2", "test3_min": "Test 3"}
    breaks = {"test1_break_min": "Before\ntest 1", "test2_break_min": "Before\ntest 2",
              "test3_break_min": "Before\ntest 3"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [6, 3, 1.3]})
    for ax, cols, label in ((axes[0], parts, "Minutes in block"), (axes[1], breaks, "Minutes on break"),
                            (axes[2], {"total_min": "Total"}, "Minutes")):
        long = df.melt(id_vars=["participant", "excluded"], value_vars=list(cols), var_name="part", value_name="min")
        long["part"] = long["part"].map(cols)
        order = list(cols.values())
        sns.boxplot(data=long, x="part", y="min", order=order, ax=ax, color=BOX_COLOR, width=0.55, fliersize=0,
                    boxprops={"alpha": 0.45}, linecolor=BOX_COLOR, linewidth=1.2)
        participant_lines(ax, long, "part", "min", order)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_ylim(0, None)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, fontsize=11)
    save(fig, path, tight_rect=(0, 0, 1, 0.93))


def main():
    args = parser(__doc__).parse_args()
    apply_dark_theme()
    study = Study.from_args(args)

    info = []
    for f in session_files():
        if load_file(f)[0] is not None:
            df = pd.read_csv(f, low_memory=False)
            if df["part"].eq("final").any():         # finished sessions; screen-outs: B06_sessions.py
                info.append(session_info(df))
    ex = study.exclusion_table.set_index("participant")
    df = pd.DataFrame(info).merge(performance(study.trials, ex), on="participant", how="left")
    demo = demographics()
    if demo is not None:
        df = df.merge(demo, on="participant", how="left")
        missing = df.loc[df["age"].isna(), "participant"].str[:8].tolist() if "age" in df else []
        if missing:
            print(f"not in the Prolific export: {', '.join(missing)}")
    else:
        print(f"no Prolific export in {PROLIFIC_DIR}: demographics left out")

    for version, dv in df.groupby("version"):
        out = study.out(f"average_version_{version}", "sanity_tests")
        dv = dv.sort_values("started_at")
        dv.to_csv(out / "participants.csv", index=False)
        title = f"{len(dv)} participants (version {version})"
        write_md(dv, out / "participants.md", f"Participants, version {version}")
        plot_timing(dv, f"Time per part of the session, {title}; red: excluded", out / "timing.png")
        print(f"version {version}: {title} -> {out}/participants.csv, participants.md, timing.png")


if __name__ == "__main__":
    main()
