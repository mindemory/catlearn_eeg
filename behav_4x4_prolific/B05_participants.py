"""Per-participant details: session, setup, time per block, practice, attention, bonus,
performance and (if a Prolific export is there) demographics.

From each kept session's CSV (load_data: task_version 1.x, not debug, with trials):
  session      task version, start (UTC), browser, OS, screen, refresh rate
  setup        viewing distance (virtual chinrest), calibration attempts, deg per px,
               comprehension-quiz attempts; minutes from the start to the first block
  time         minutes in each block (end of its intro screen -> last trial) and on the
               break before it (last trial of the previous block -> end of this block's
               intro: summary, screen check, intro), and in total
  practice     trials to criterion and accuracy, per practice block
  test         accuracy, late answers and bonus per test block; P(F) and the key-bias
               exclusion (load_data.exclusions); whether accuracy over all test trials is
               above chance (one-sided binomial, p < .05)
  attention    window/tab switches and fullscreen exits
  bonus        what the task promised (its final bonus_usd), per block and in total
Demographics: Prolific's export ("Download demographic data" on the study's Submissions
page), saved as any .csv in ROOT/prolific/; merged on the participant ID. Without it the
demographic columns are left out.

Outputs, in ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/analysis/average_version_<A|B>/sanity_tests/:
  participants.csv   every value, one row per participant
  participants.md    the same as readable tables
  timing.png         minutes per part of the session (box plots over participants; red:
                     excluded)

  ~/miniforge3/envs/kernelbehav/bin/python B05_participants.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from B03_screening import BOX, participant_lines  # noqa: E402
from load_data import DATA_DIR, OUT_DIR, ROOT, exclusions, load_file, load_trials  # noqa: E402
from style import apply_dark_theme  # noqa: E402

PROLIFIC_DIR = ROOT / "prolific"
TRIAL_PARTS = ("iti", "response", "feedback")
RULE_TYPE = lambda rule: rule.split("_")[1]   # noqa: E731   'x4_VI_A' -> 'VI'


def num(s):
    return pd.to_numeric(s, errors="coerce")


def first(df, col):
    v = df[col].dropna() if col in df else pd.Series(dtype=object)
    return v.iloc[0] if len(v) else np.nan


def last(df, col):
    v = df[col].dropna() if col in df else pd.Series(dtype=object)
    return v.iloc[-1] if len(v) else np.nan


def session_info(df):
    """One participant's details from the raw session CSV."""
    df = df.reset_index(drop=True)
    t = num(df["time_elapsed"])
    minutes = lambda ms: ms / 60000   # noqa: E731
    pid = str(first(df, "prolific_pid"))
    out = {
        "participant": pid, "version": first(df, "version"), "task_version": first(df, "task_version"),
        "run": "main" if "iti_ms" in df.columns else "pilot",
        "file": first(df, "session_file"), "started_at": first(df, "started_at"),
        "finished": df["part"].eq("final").any(),
        "browser": f"{first(df, 'browser')} {first(df, 'browser_version')}", "os": first(df, "os"),
        "screen": f"{num(first(df, 'width')):.0f} x {num(first(df, 'height')):.0f}",
        "refresh_hz": round(num(first(df, "vsync_rate")), 1),
        "view_dist_cm": round(num(first(df, "view_dist_mm")) / 10, 1),
        "calibration_attempts": num(last(df, "calibration_attempts")),
        "px_per_deg": round(num(last(df, "px_per_deg")), 1),
        "quiz_attempts": int(df["part"].eq("comprehension").sum()),
        "blur_count": num(df.get("blur_count")).max(), "fullscreen_exits": num(df.get("fullscreen_exit_count")).max(),
    }
    trial_rows = df.index[df["part"].isin(TRIAL_PARTS)]
    out["setup_min"] = round(minutes(t[trial_rows[0] - 1]), 1) if len(trial_rows) else np.nan

    summaries = df[df["part"] == "block_summary"]
    prev_end = None
    test_k = practice_k = 0
    bonus = json.loads(last(df, "block_bonuses_usd")) if isinstance(last(df, "block_bonuses_usd"), str) else []
    for _, s in summaries.iterrows():
        blk = s["block"]
        rows = df.index[df["part"].isin(TRIAL_PARTS) & (num(df["block"]) == num(blk))]
        start, end = t[rows[0] - 1], t[rows[-1]]
        rule = df.loc[rows[0], "rule"]
        if s["phase"] == "practice":
            practice_k += 1
            key = f"practice{practice_k}"
            out[f"{key}_rule"] = RULE_TYPE(rule)
            out[f"{key}_trials"] = int(num(s["block_rounds"]))
            out[f"{key}_acc"] = round(num(s["block_pcorrect"]), 3)
            out[f"{key}_criterion_met"] = str(s["practice_criterion_met"]).lower() == "true"
        else:
            test_k += 1
            key = f"test{test_k}"
            out[f"{key}_rule"] = RULE_TYPE(rule)
            out[f"{key}_acc"] = round(num(s["block_pcorrect"]), 3)
            out[f"{key}_late"] = int(num(s["block_n_late"]))
            out[f"{key}_bonus_usd"] = num(s["block_bonus_usd"])
        out[f"{key}_min"] = round(minutes(end - start), 1)
        if prev_end is not None:
            out[f"{key}_break_min"] = round(minutes(start - prev_end), 1)
        prev_end = end
    out["total_min"] = round(minutes(t.max()), 1)
    out["bonus_usd"] = num(last(df, "bonus_usd"))
    if bonus and len(bonus) != test_k:
        print(f"{pid[:8]}: {len(bonus)} block bonuses for {test_k} test blocks")
    return out


def performance(trials):
    """Test accuracy over all blocks, binomial test against chance, P(F) and exclusion."""
    ex = exclusions(trials).set_index("participant")
    rows = []
    for pid, r in trials[trials["phase"] == "test"].groupby("participant", sort=False):
        k, n = int(r["correct"].sum()), len(r)
        rows.append({"participant": pid, "test_acc": round(k / n, 3),
                     "test_above_chance_p": binomtest(k, n, 0.5, alternative="greater").pvalue,
                     "p_f": round(ex.loc[pid, "p_f"], 3), "excluded": bool(ex.loc[pid, "excluded"])})
    out = pd.DataFrame(rows)
    out["above_chance"] = out["test_above_chance_p"] < 0.05
    return out


def demographics():
    """Prolific's demographic exports in ROOT/prolific/, one row per participant, or None."""
    files = sorted(PROLIFIC_DIR.glob("*.csv")) if PROLIFIC_DIR.is_dir() else []
    if not files:
        return None
    d = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    d = d.rename(columns={"participant_id": "participant", "status": "prolific_status",
                          "time_taken": "prolific_time_taken_s", "started_at": "prolific_started_at",
                          "completed_at": "prolific_completed_at"})
    drop = [c for c in d.columns if c in ("submission_id", "completion_code", "reviewed_at", "archived_at",
                                           "custom_study_tncs_accepted_at")]
    return d.drop(columns=drop).drop_duplicates("participant", keep="last")


# ---------------------------------------------------------------- outputs
def md_table(df):
    cols = list(df.columns)
    fmt = lambda v: "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_md(df, path, title):
    df = df.assign(id=df["participant"].str[:8])
    sections = {
        "Session": ["id", "run", "task_version", "started_at", "browser", "os", "screen", "refresh_hz"],
        "Setup and attention": ["id", "view_dist_cm", "calibration_attempts", "px_per_deg", "quiz_attempts",
                                "blur_count", "fullscreen_exits"],
        "Time (minutes)": ["id", "setup_min", "practice1_min", "practice2_min", "test1_break_min", "test1_min",
                           "test2_break_min", "test2_min", "test3_break_min", "test3_min", "total_min"],
        "Practice": ["id", "practice1_rule", "practice1_trials", "practice1_acc", "practice2_rule", "practice2_trials",
                     "practice2_acc"],
        "Test and bonus": ["id", "test1_acc", "test2_acc", "test3_acc", "test_acc", "above_chance", "p_f", "excluded",
                           "test1_bonus_usd", "test2_bonus_usd", "test3_bonus_usd", "bonus_usd"],
    }
    demo = [c for c in ("age", "sex", "ethnicity_simplified", "country_of_residence", "nationality", "language",
                        "fluent_languages", "student_status", "employment_status", "prolific_status",
                        "prolific_time_taken_s") if c in df]
    if demo:
        sections["Demographics (Prolific)"] = ["id"] + demo
    parts = [f"# {title}", ""]
    for name, cols in sections.items():
        parts += [f"## {name}", "", md_table(df[[c for c in cols if c in df]]), ""]
    if not demo:
        parts += ["Demographics: no Prolific export found in " + str(PROLIFIC_DIR) + ".", ""]
    total = df["bonus_usd"].sum()
    parts += [f"Bonus promised in total: ${total:.2f} ({len(df)} participants).", ""]
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
        sns.boxplot(data=long, x="part", y="min", order=order, ax=ax, color=BOX, width=0.55, fliersize=0,
                    boxprops={"alpha": 0.45}, linecolor=BOX, linewidth=1.2)
        participant_lines(ax, long, "part", "min", order)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_ylim(0, None)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-sync", action="store_true", help="don't copy new files from the Drive folder")
    args = parser.parse_args()
    apply_dark_theme()

    trials = load_trials(sync=not args.no_sync)
    info = []
    for f in sorted(DATA_DIR.glob("catlearn_online_*.csv")):
        if load_file(f)[0] is not None:
            info.append(session_info(pd.read_csv(f, low_memory=False)))
    df = pd.DataFrame(info).merge(performance(trials), on="participant", how="left")
    demo = demographics()
    if demo is not None:
        df = df.merge(demo, on="participant", how="left")
        missing = df.loc[df["age"].isna(), "participant"].str[:8].tolist() if "age" in df else []
        if missing:
            print(f"not in the Prolific export: {', '.join(missing)}")
    else:
        print(f"no Prolific export in {PROLIFIC_DIR}: demographics left out")

    for version, dv in df.groupby("version"):
        out = OUT_DIR / f"average_version_{version}" / "sanity_tests"
        out.mkdir(parents=True, exist_ok=True)
        dv = dv.sort_values("started_at")
        dv.to_csv(out / "participants.csv", index=False)
        title = f"{len(dv)} participants (version {version})"
        write_md(dv, out / "participants.md", f"Participants, version {version}")
        plot_timing(dv, f"Time per part of the session, {title}; red: excluded", out / "timing.png")
        print(f"version {version}: {title} -> {out}/participants.csv, participants.md, timing.png")


if __name__ == "__main__":
    main()
