"""Load the F/J category-learning study's data (catlearn_4x4_prolific, task_version 1.x).

Files arrive in the Google Drive folder DataPipe writes to (DRIVE_DIR). load_trials() first
copies new or changed finished sessions (catlearn_online_*.csv) into DATA_DIR (sync=False
skips this; files archived in DATA_DIR/old_versions/ are never copied again), then reads
them and returns one row per trial (part == 'response'). Unfinished sessions (only a
.partial.json on Drive) are not copied.

Folders, in ROOT (~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/):
  data/       the finished sessions' CSVs (DATA_DIR)
  analysis/   figures and tables (OUT_DIR)
  bonus/      the Prolific bonus list (catlearn_4x4_prolific/tools/bonus_payments.py)

Kept: task_version 1.x (the F/J study), not debug runs, and sessions with at least one
trial. Left out, and listed: sessions without trials (e.g. excluded by the browser check),
the slot-machine pilot builds (3.x).

Trial columns: participant, version (A / B), block, phase ('practice' / 'test'), rule,
trial_in_block, rep (pass through the pairs), compound, level_a, level_b, category
(1 = F, 0 = J), choice (1 = pressed F, 0 = J, NaN = late), correct, timeout, rt (ms,
NaN = late).
"""

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "Documents" / "data" / "catlearn_eeg" / "catlearn_4x4_prolific"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "analysis"
DRIVE_DIR = Path.home() / "My Drive" / "DataPipe" / "catlearn_4x4"
PATTERN = "catlearn_online_*.csv"          # finished sessions only (not *.partial.json)

COLUMNS = ["participant", "version", "block", "phase", "rule", "trial_in_block", "rep", "compound", "level_a",
           "level_b", "category", "choice", "correct", "timeout", "rt"]


def sync_from_drive(src=DRIVE_DIR, dst=DATA_DIR):
    """Copy finished sessions that are new or changed in the Drive folder; returns the names copied."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        print(f"Drive folder not found ({src}); using the files already in {dst}")
        return []
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(src.glob(PATTERN)):
        if (dst / "old_versions" / f.name).exists():    # archived on purpose: don't bring it back
            continue
        target = dst / f.name
        if not target.exists() or target.stat().st_size != f.stat().st_size or target.stat().st_mtime < f.stat().st_mtime:
            shutil.copy2(f, target)
            copied.append(f.name)
    return copied


def _bool(s):
    return s.astype(str).str.lower().eq("true")


def load_file(path):
    """(trials, None) for a kept session, or (None, why it is left out)."""
    df = pd.read_csv(path, low_memory=False)
    first = df.iloc[0]
    version = str(first.get("task_version", ""))
    if not version.startswith("1."):
        return None, f"task_version {version}"
    if str(first.get("debug")).lower() == "true":
        return None, "debug run"
    r = df[df["part"] == "response"]
    if r.empty:
        return None, "no trials"
    pid = df["prolific_pid"].dropna().astype(str)
    pid = pid[pid != ""]
    r = r.reindex(columns=COLUMNS[2:]).copy()
    r.insert(0, "version", first.get("version"))
    r.insert(0, "participant", pid.iloc[0] if len(pid) else path.stem.replace("catlearn_online_", ""))
    for c in ("correct", "timeout"):
        r[c] = _bool(r[c])
    for c in ("block", "trial_in_block", "rep", "compound", "level_a", "level_b", "category"):
        r[c] = r[c].astype(int)
    r["choice"] = pd.to_numeric(r["choice"], errors="coerce")
    r["rt"] = pd.to_numeric(r["rt"], errors="coerce")
    return r, None


def load_trials(data_dir=DATA_DIR, sync=True):
    """One row per trial of every kept session (see the module docstring)."""
    if sync:
        copied = sync_from_drive(DRIVE_DIR, data_dir)
        print(f"copied {len(copied)} new or changed file(s) from {DRIVE_DIR}")
    data_dir = Path(data_dir)
    kept, skipped = [], []
    for f in sorted(data_dir.glob("catlearn_online_*.csv")):
        r, why = load_file(f)
        if r is None:
            skipped.append(f"{f.name} ({why})")
        else:
            kept.append(r)
    if skipped:
        print(f"left out {len(skipped)} file(s): " + "; ".join(skipped))
    if not kept:
        raise FileNotFoundError(f"no F/J-study sessions with trials in {data_dir}")
    return pd.concat(kept, ignore_index=True)


# ---------------------------------------------------------------- exclusions
# Key bias: P(F) over all answered test trials outside [BIAS_LIMITS]. Every rule has 8 F and
# 8 J pairs, so a participant who mostly presses one key isn't discriminating the pairs.
BIAS_LIMITS = (0.25, 0.75)


def exclusions(trials, limits=BIAS_LIMITS):
    """One row per participant: P(F) over the answered test trials, and whether (and why) the
    participant is excluded from the averages."""
    t = trials[(trials["phase"] == "test") & ~trials["timeout"]]
    p_f = t.groupby("participant", sort=False)["choice"].apply(lambda c: c.eq(1).mean())
    out = p_f.rename("p_f").reset_index()
    out["excluded"] = (out["p_f"] < limits[0]) | (out["p_f"] > limits[1])
    out["reason"] = out["excluded"].map({True: f"key bias: P(F) outside {limits[0]}-{limits[1]}", False: ""})
    return out


def save_exclusions(trials, out_dir=OUT_DIR):
    """exclusions.csv in the analysis folder; returns the set of excluded participants."""
    ex = exclusions(trials)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex.to_csv(out_dir / "exclusions.csv", index=False)
    excluded = set(ex.loc[ex["excluded"], "participant"])
    if excluded:
        print(f"excluded from averages ({len(excluded)}): "
              + ", ".join(f"{p[:8]} (P(F) {v:.2f})" for p, v in ex.loc[ex["excluded"], ["participant", "p_f"]].values))
    return excluded
