"""Load the F/J category-learning study's data (catlearn_4x4_prolific, task_version 1.x).

Files arrive in the Google Drive folder DataPipe writes to (params.DRIVE_DIR). load_trials()
first copies new or changed finished sessions (catlearn_online_*.csv) into params.DATA_DIR
(sync=False skips this; files archived in params.ARCHIVES are never copied again), then
reads them and returns one row per trial (part == 'response'). Unfinished sessions (only a
.partial.json on Drive) are not copied.

Kept: task_version 1.x, not debug runs, and sessions with at least one trial; and for a
participant with several sessions (e.g. someone who took both the A and the B study), only the
first: the later ones come from someone who already knew the task. Left out, and listed:
sessions without trials (e.g. screened out at the quiz or the browser check), repeat
sessions, the slot-machine builds (3.x, archived in data/old_versions/).

Trial columns (params.TRIAL_COLUMNS): participant, version (A / B), block, phase ('practice' / 'test'), rule, trial_in_block, rep (pass
through the pairs), compound, level_a, level_b, fractal_a, fractal_b (the task's fractal files
shown left and right), category (1 = F, 0 = J), choice (1 = pressed
F, 0 = J, NaN = late), correct, timeout, rt (ms, NaN = late).

Exclusions (exclusions()): key bias and rushing, automatic, and the participants listed by
hand in params.MANUAL_EXCLUSIONS.
"""

import shutil
from pathlib import Path

import pandas as pd

from scipy.stats import binomtest

from params import (ARCHIVES, BIAS_LIMITS, DATA_DIR, DRIVE_DIR, FAST_RT_MS, MANUAL_EXCLUSIONS, RUSH_CHANCE_ALPHA,
                    RUSH_FAST_SHARE, SESSION_PATTERN, TASK_VERSION_PREFIX, TRIAL_COLUMNS)


# ---------------------------------------------------------------- files
def sync_from_drive(src=DRIVE_DIR, dst=DATA_DIR):
    """Copy finished sessions that are new or changed in the Drive folder; returns the names copied."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        print(f"Drive folder not found ({src}); using the files already in {dst}")
        return []
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(src.glob(SESSION_PATTERN)):
        if any((a / f.name).exists() for a in ARCHIVES):   # archived on purpose: don't bring it back
            continue
        target = dst / f.name
        if not target.exists() or target.stat().st_size != f.stat().st_size or target.stat().st_mtime < f.stat().st_mtime:
            shutil.copy2(f, target)
            copied.append(f.name)
    return copied


def session_files(data_dir=DATA_DIR):
    """The session CSVs to analyse."""
    return sorted(Path(data_dir).glob(SESSION_PATTERN))


def _bool(s):
    return s.astype(str).str.lower().eq("true")


def load_file(path):
    """(trials, None) for a kept session, or (None, why it is left out)."""
    df = pd.read_csv(path, low_memory=False)
    first = df.iloc[0]
    version = str(first.get("task_version", ""))
    if not version.startswith(TASK_VERSION_PREFIX):
        return None, f"task_version {version}"
    if str(first.get("debug")).lower() == "true":
        return None, "debug run"
    r = df[df["part"] == "response"]
    if r.empty:
        return None, "no trials"
    pid = df["prolific_pid"].dropna().astype(str)
    pid = pid[pid != ""]
    r = r.reindex(columns=TRIAL_COLUMNS[2:]).copy()
    r.insert(0, "version", first.get("version"))
    r.insert(0, "participant", pid.iloc[0] if len(pid) else path.stem.replace("catlearn_online_", ""))
    for c in ("correct", "timeout"):
        r[c] = _bool(r[c])
    for c in ("block", "trial_in_block", "rep", "compound", "level_a", "level_b", "fractal_a", "fractal_b", "category"):
        r[c] = r[c].astype(int)
    r["choice"] = pd.to_numeric(r["choice"], errors="coerce")
    r["rt"] = pd.to_numeric(r["rt"], errors="coerce")
    return r, None


def load_trials(data_dir=DATA_DIR, sync=True):
    """One row per trial of every kept session (see the module docstring)."""
    if sync:
        copied = sync_from_drive(DRIVE_DIR, data_dir)
        print(f"copied {len(copied)} new or changed file(s) from {DRIVE_DIR}")
    kept, skipped, first_file = [], [], {}
    for f in session_files(data_dir):              # file names start with the start time: in order
        r, why = load_file(f)
        pid = None if r is None else r["participant"].iloc[0]
        if r is not None and pid in first_file:
            why, r = f"repeat session of {pid[:8]}, who first took part in {first_file[pid]}", None
        if r is None:
            skipped.append(f"{f.name} ({why})")
        else:
            first_file[pid] = f.name
            kept.append(r)
    if skipped:
        print(f"left out {len(skipped)} file(s): " + "; ".join(skipped))
    if not kept:
        raise FileNotFoundError(f"no F/J-study sessions with trials in {data_dir}")
    return pd.concat(kept, ignore_index=True)


# ---------------------------------------------------------------- exclusions
def manual_exclusions(path=MANUAL_EXCLUSIONS):
    """Exclusions decided by hand: participant -> (reason, decided_on)."""
    if not Path(path).exists():
        return {}
    m = pd.read_csv(path, dtype=str).fillna("")
    return {r.participant: (r.reason, r.decided_on) for r in m.itertuples()}


def exclusions(trials, limits=BIAS_LIMITS):
    """One row per participant: version, P(F) over the answered test trials, and whether
    (and why) the participant is excluded from the averages. Three kinds:
      key_bias  automatic: P(F) outside `limits` (every rule has 8 F and 8 J pairs, so a
                participant who mostly presses one key isn't discriminating the pairs)
      rushing   automatic: in some test block, at least params.RUSH_FAST_SHARE of the answers
                faster than params.FAST_RT_MS with that block's accuracy at chance, and accuracy
                over all test trials at chance too ('at chance': one-sided binomial p >=
                params.RUSH_CHANCE_ALPHA): answering without looking. Someone who rushed one
                block but learned the others is kept.
      manual    listed in params.MANUAL_EXCLUSIONS, with the reason and the date decided"""
    t = trials[(trials["phase"] == "test") & ~trials["timeout"]]
    info = trials.groupby("participant", sort=False)[["version"]].first()
    p_f = t.groupby("participant", sort=False)["choice"].apply(lambda c: c.eq(1).mean())
    out = info.join(p_f.rename("p_f"), how="inner").reset_index()
    manual = manual_exclusions()
    rushed = rushing(trials)
    rules, reasons, dates = [], [], []
    for pid, pf in zip(out["participant"], out["p_f"]):
        rule, reason, date = [], [], []
        if pf < limits[0] or pf > limits[1]:
            rule.append("key_bias")
            reason.append(f"key bias: pressed F on {pf:.0%} of the answered test trials "
                          f"(allowed {limits[0]:.0%}-{limits[1]:.0%}; every rule has 8 F and 8 J pairs)")
        if pid in rushed:
            rule.append("rushing")
            reason.append("rushing: " + "; ".join(rushed[pid]))
        if pid in manual:
            rule.append("manual")
            reason.append(manual[pid][0])
            date.append(manual[pid][1])
        rules.append("; ".join(rule))
        reasons.append(" | ".join(reason))
        dates.append("; ".join(date))
    out["excluded"] = [bool(r) for r in rules]
    out["rule"] = rules
    out["reason"] = reasons
    out["decided_on"] = dates
    missing = set(manual) - set(out["participant"])
    if missing:
        print(f"manual exclusions not in the data analysed: {', '.join(p[:8] for p in sorted(missing))}")
    return out


def at_chance(b, alpha=RUSH_CHANCE_ALPHA):
    """Accuracy of a set of trials not above 0.5 (one-sided binomial p >= alpha; late = wrong)."""
    return binomtest(int(b["correct"].sum()), len(b), 0.5, alternative="greater").pvalue >= alpha


def rushing(trials, fast=FAST_RT_MS, share=RUSH_FAST_SHARE, alpha=RUSH_CHANCE_ALPHA):
    """{participant: [block descriptions]} for participants at chance over all test trials with
    at least one test block of `share` or more answers faster than `fast` ms at chance."""
    out = {}
    test = trials[trials["phase"] == "test"]
    overall = {pid: at_chance(r, alpha) for pid, r in test.groupby("participant", sort=False)}
    for (pid, block), b in test.groupby(["participant", "block"], sort=False):
        answered = b[~b["timeout"]]
        k, n = int(b["correct"].sum()), len(b)
        fast_share = (answered["rt"] < fast).mean() if len(answered) else 0.0
        if overall[pid] and fast_share >= share and at_chance(b, alpha):
            rule = b["rule"].iloc[0]
            out.setdefault(pid, []).append(f"{rule.split('_')[1]}: {fast_share:.0%} of answers under {fast} ms, "
                                           f"accuracy {k / n:.2f}")
    return out
