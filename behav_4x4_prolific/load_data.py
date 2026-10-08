"""Load the slot-machine experiment's data files (catlearn_4x4_prolific) into tidy tables.

Each participant's CSV (one row per screen, see catlearn_4x4_prolific/README.md) becomes:
  rounds    one row per round (part == 'response'), with the signal-detection outcome
  sessions  one row per participant: design, calibration, comprehension, bonus

Signal detection: a category-1 pair is the signal and answering category 1 is the
positive answer. In task_version 1.x (the F/J study) that is an F pair and an F press; in
the slot-machine pilot builds (3.x) a "win" pair and a YES press.
  hit  = win pair, answered YES         miss = win pair, answered NO or late
  fa   = lose pair, answered YES        cr   = lose pair, answered NO or late
Late rounds never contain a YES, so they count as misses (win pairs) or correct
rejections (lose pairs) here, and are also counted separately (timeout).

Files arrive in the Google Drive folder DataPipe writes to (DRIVE_DIR). load_all() first
copies new or changed ones into DATA_DIR (sync=False skips this; files archived in
DATA_DIR/old_versions/ are never copied again). Finished sessions are CSVs; a session that
never finished leaves only a .partial.json, which is copied but not analysed here.
"""

import shutil
from pathlib import Path

import pandas as pd

DATA_DIR = Path.home() / "Documents" / "data" / "catlearn_eeg" / "catlearn_4x4_prolific"
OUT_DIR = DATA_DIR / "analysis"
DRIVE_DIR = Path.home() / "My Drive" / "DataPipe" / "catlearn_4x4"
PATTERNS = ("catlearn_online_*.csv", "catlearn_online_*.partial.json")

ROUND_COLUMNS = ["block", "phase", "block_in_phase", "rule", "rule_type", "size", "trial_in_block", "rep",
                 "compound", "level_a", "level_b", "fractal_a", "fractal_b", "category", "response", "choice",
                 "rt", "correct", "timeout", "coins_delta", "round_started_by", "wait_ms", "blur_count",
                 "fullscreen_exit_count", "time_elapsed"]


def participant_id(df, path):
    pid = df["prolific_pid"].dropna().astype(str)
    pid = pid[pid != ""]
    return pid.iloc[0] if len(pid) else path.stem.replace("catlearn_online_", "")


def _bool(s):
    return s.astype(str).str.lower().eq("true")


def load_file(path):
    df = pd.read_csv(path, low_memory=False)
    pid = participant_id(df, path)
    # columns a task version doesn't record (e.g. coins in the F/J study) are left empty
    r = df[df["part"] == "response"].reindex(columns=ROUND_COLUMNS).copy()
    for c in ("correct", "timeout"):
        r[c] = _bool(r[c])
    for c in ("block", "block_in_phase", "size", "trial_in_block", "rep", "compound", "level_a", "level_b",
              "category"):
        r[c] = r[c].astype(int)
    r["yes"] = r["choice"].eq(1)
    win = r["category"].eq(1)
    r["hit"] = win & r["yes"]
    r["miss"] = win & ~r["yes"]
    r["fa"] = ~win & r["yes"]
    r["cr"] = ~win & ~r["yes"]
    r.insert(0, "participant", pid)

    first = df[df["part"] == "response"].iloc[0] if len(r) else df.iloc[0]
    cal = df[df["part"] == "calibration_summary"]
    quiz = df[df["part"] == "comprehension"]
    final = df[df["part"] == "final"]
    summaries = df[df["part"] == "block_summary"]
    session = {
        "participant": pid,
        "file": path.name,
        "debug": str(first.get("debug")).lower() == "true",
        "task_version": first.get("task_version"),
        "seed": first.get("seed"),
        "test_type": first.get("test_type"),             # versions 3.x
        "test_first": first.get("test_first"),
        "key_yes": first.get("key_yes"),
        "version": first.get("version"),                 # F/J study: A or B, and the test types in order
        "sequence": first.get("sequence"),
        "calibration_source": cal["calibration_source"].iloc[-1] if len(cal) else None,
        "view_dist_mm": cal["view_dist_mm"].iloc[-1] if len(cal) else None,
        "px_per_deg": cal["px_per_deg"].iloc[-1] if len(cal) else None,
        "comprehension_passed": bool(_bool(quiz["comprehension_passed"]).iloc[-1]) if len(quiz) else None,
        "comprehension_attempts": len(quiz),
        "completed": len(final) > 0,
        "bonus_coins": final["bonus_coins"].iloc[0] if len(final) and "bonus_coins" in final else None,
        "bonus_usd": final["bonus_usd"].iloc[0] if len(final) else None,
        "practice_criterion_met": ";".join(summaries.loc[summaries["phase"] == "practice", "practice_criterion_met"]
                                           .astype(str).tolist()),
        "n_rounds": len(r),
        "max_blur_count": r["blur_count"].max() if len(r) else None,
    }
    return r, session


def sync_from_drive(src=DRIVE_DIR, dst=DATA_DIR):
    """Copy data files that are new or changed in the Drive folder; returns the names copied"""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        print(f"Drive folder not found ({src}); using the files already in {dst}")
        return []
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in PATTERNS:
        for f in sorted(src.glob(pattern)):
            if (dst / "old_versions" / f.name).exists():    # archived on purpose: don't bring it back
                continue
            target = dst / f.name
            if not target.exists() or target.stat().st_size != f.stat().st_size or target.stat().st_mtime < f.stat().st_mtime:
                shutil.copy2(f, target)
                copied.append(f.name)
    return copied


def load_all(data_dir=DATA_DIR, sync=True, drive_dir=DRIVE_DIR):
    if sync:
        copied = sync_from_drive(drive_dir, data_dir)
        print(f"copied {len(copied)} new or changed file(s) from {drive_dir}" + (f": {', '.join(copied)}" if copied else ""))
    partial = [f.name for f in Path(data_dir).glob("catlearn_online_*.partial.json")
               if not (Path(data_dir) / (f.name.split(".partial.json")[0].rsplit("-", 1)[0] + ".csv")).exists()]
    if partial:
        print(f"{len(partial)} unfinished session(s) (partial files, not analysed): {', '.join(sorted(partial))}")
    files = sorted(Path(data_dir).glob("catlearn_online_*.csv"))
    if not files:
        raise FileNotFoundError(f"no catlearn_online_*.csv in {data_dir}")
    rounds, sessions = zip(*(load_file(f) for f in files))
    return pd.concat(rounds, ignore_index=True), pd.DataFrame(sessions)


def sdt_rates(r):
    """Accuracy, hit / miss / false-alarm rates and F1 of a set of rounds (dict)."""
    n_win, n_lose = int((r["category"] == 1).sum()), int((r["category"] == 0).sum())
    hits, misses, fas = int(r["hit"].sum()), int(r["miss"].sum()), int(r["fa"].sum())
    nan = float("nan")
    denom_f1 = 2 * hits + misses + fas
    return {
        "n": len(r),
        "accuracy": r["correct"].mean() if len(r) else nan,
        "hit_rate": hits / n_win if n_win else nan,
        "miss_rate": misses / n_win if n_win else nan,
        "fa_rate": fas / n_lose if n_lose else nan,
        "f1": 2 * hits / denom_f1 if denom_f1 else nan,
        "late_rate": r["timeout"].mean() if len(r) else nan,
        "rt_median": r.loc[~r["timeout"], "rt"].median(),
    }
