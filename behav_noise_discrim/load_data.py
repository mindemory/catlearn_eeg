"""Load the noise discrimination study's data files (noise_discrim_prolific) into tidy tables.

Files arrive in the Google Drive folder DataPipe writes to (DRIVE_DIR); sync_from_drive()
copies new or changed ones into DATA_DIR, which the analysis reads. Two kinds of file:
  <name>.csv            a finished session (the whole session, uploaded at the end)
  <name>...partial.json  trials DataPipe staged for a session that never finished (the
                         participant quit); used only when that session has no .csv

Each participant's CSV (one row per screen, see noise_discrim_prolific/README.md) becomes:
  trials    one row per main trial (part == 'response'; practice trials are left out)
  sessions  one row per participant: calibration, comprehension, timing quality

Signal detection: a "different" pair is the signal and "different" is the positive answer.
  hit = different pair, answered different     fa = same pair, answered different
Late trials (no answer within 2 s) have no answer; they are left out of the rates and
counted separately.

Timing: the patches are switched on and off in display frames (stim_hidden_by == 'frame').
If the browser stopped drawing frames (stim_hidden_by == 'timer') the exposure is unknown,
so those trials are flagged (timing_ok = False); so are exposures more than half a 60 Hz
frame from the target (TIMING.stimulus, from the file's design row: 100 ms from version
1.1, 200 ms in 1.0). Half a frame keeps the nearest frame count on 75 or 144 Hz screens
(107 / 97 ms) and drops a patch left on one frame too long (117 ms at 60 Hz).
"""

import json
import shutil
from pathlib import Path

import pandas as pd

DATA_DIR = Path.home() / "Documents" / "data" / "catlearn_eeg" / "noise_discrim_prolific"
DRIVE_DIR = Path.home() / "My Drive" / "DataPipe" / "noise_discrim_thresholding"
PATTERNS = ("noise_discrim_*.csv", "noise_discrim_*.partial.json")
OUT_DIR = DATA_DIR / "analysis"
TOLERANCE_MS = 0.5 * 1000 / 60

TRIAL_COLUMNS = ["block", "trial_in_block", "alpha", "alpha_level", "pair", "seed_left", "seed_right", "response",
                 "answer", "correct", "timeout", "rt", "stim_ms", "stim_frames", "stim_hidden_by",
                 "answered_during_stimulus", "blur_count", "fullscreen_exit_count", "px_per_deg", "layout_scale",
                 "time_elapsed"]


def _bool(s):
    return s.astype(str).str.lower().eq("true")


def participant_id(df, path):
    """Prolific ID; else the ?id= label of a shared link; else the file name"""
    for col in ("prolific_pid", "participant"):
        if col in df:
            v = df[col].dropna().astype(str)
            v = v[(v != "") & (v != "pilot")]
            if len(v):
                return v.iloc[0]
    return session_key(path).replace("noise_discrim_", "")


def target_ms(df):
    """The intended exposure: TIMING.stimulus from the design row (version 1.0 had no record: 200 ms)"""
    design = df[df["part"] == "design"]
    if "timing_json" in df and len(design) and isinstance(design["timing_json"].iloc[0], str):
        return json.loads(design["timing_json"].iloc[0])["stimulus"]
    return 200


def sync_from_drive(src=DRIVE_DIR, dst=DATA_DIR):
    """Copy data files that are new or changed in the Drive folder; returns the names copied.
    Files archived in dst/old_versions/ are not copied again."""
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


def session_key(path):
    """The session a file belongs to: its CSV name without .csv. DataPipe names a partial file
    <name>-<staging id>.partial.json, so the staging id is dropped too."""
    name = path.name
    if name.endswith(".partial.json"):
        return name[: -len(".partial.json")].rsplit("-", 1)[0]
    return name.removesuffix(".csv")


def read_table(path):
    """A data file as a DataFrame: a jsPsych CSV, or DataPipe's .partial.json (a list of trial rows)"""
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):   # rows under some key, e.g. {"trials": [...]}
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    return pd.DataFrame(raw)


def load_file(path):
    df = read_table(path)
    if "part" not in df or not (df["part"] == "response").any():
        return None                                      # no trials (e.g. consent declined)
    pid = participant_id(df, path)
    r = df[df["part"] == "response"]
    if "phase" in r:                                     # practice trials (version 1.3 on) are not analysed
        r = r[r["phase"].fillna("main") != "practice"]
    t = r[TRIAL_COLUMNS].copy()
    for c in ("correct", "timeout", "answered_during_stimulus"):
        t[c] = _bool(t[c])
    for c in ("block", "trial_in_block", "alpha_level"):
        t[c] = t[c].astype(int)
    t["signal"] = t["pair"].eq("different")
    t["said_different"] = t["answer"].eq("different")
    target = target_ms(df)
    t["stim_target_ms"] = target
    exposure_ok = t["stim_hidden_by"].eq("frame") & (t["stim_ms"] - target).abs().le(TOLERANCE_MS)
    t["timing_ok"] = exposure_ok | t["answered_during_stimulus"]
    t.insert(0, "participant", pid)

    cal = df[df["part"] == "calibration_summary"]
    quiz = df[df["part"] == "comprehension"]
    final = df[df["part"] == "final"]
    session = {
        "participant": pid,
        "file": path.name,
        "seed": df["seed"].dropna().iloc[0] if "seed" in df else None,
        "key_same": df["key_same"].dropna().iloc[0] if "key_same" in df else None,
        "debug": _bool(df["debug"]).any() if "debug" in df else False,
        "simulated": _bool(df["simulated"]).any() if "simulated" in df else False,
        "finished": len(final) > 0,
        "partial": path.name.endswith(".partial.json"),
        "n_trials": len(t),
        "pcorrect": t["correct"].mean(),
        "late_rate": t["timeout"].mean(),
        "task_version": df["task_version"].dropna().iloc[0] if "task_version" in df else None,
        "stim_target_ms": target,
        "stim_ms_median": t["stim_ms"].median(),
        "timing_bad_rate": 1 - t["timing_ok"].mean(),
        "calibration_source": cal["calibration_source"].iloc[-1] if len(cal) else None,
        "px_per_deg": cal["px_per_deg"].iloc[-1] if len(cal) else None,
        "layout_scale": cal["layout_scale"].iloc[-1] if len(cal) else None,
        "comprehension_passed": _bool(quiz["comprehension_passed"]).iloc[-1] if len(quiz) else None,
        "blur_count": t["blur_count"].max(),
        "fullscreen_exit_count": t["fullscreen_exit_count"].max(),
    }
    return t, session


MIN_VERSION = "1.3.0"   # the current design: practice + 8 blocks of 80 trials, 100 ms


def _version(v):
    return tuple(int(x) for x in str(v).split(".")) if v else (0,)


def load_all(data_dir=DATA_DIR, include_debug=False, include_partial=False, min_version=MIN_VERSION):
    """Every data file in data_dir (not its subfolders), keeping complete sessions of task version
    >= min_version. Left out unless asked: debug runs (?debug=1, shortened) and unfinished
    sessions (.partial.json, or a CSV without the final screen)."""
    files = sorted(f for pattern in PATTERNS for f in Path(data_dir).glob(pattern))
    finished = {session_key(f) for f in files if f.suffix == ".csv"}
    files = [f for f in files if f.suffix == ".csv" or session_key(f) not in finished]   # partial only if no .csv
    if not files:
        raise SystemExit(f"no noise_discrim_* data files in {data_dir}")
    loaded = [x for x in (load_file(f) for f in files) if x is not None]
    rules = [("older task version", lambda s: _version(s["task_version"]) < _version(min_version), "--min-version"),
             ("debug run", lambda s: s["debug"], "--include-debug")]
    if not include_partial:
        rules.append(("unfinished session", lambda s: s["partial"] or not s["finished"], "--include-partial"))
    for label, drop, flag in rules:
        if label == "debug run" and include_debug:
            continue
        skipped = [s["file"] for _, s in loaded if drop(s)]
        if skipped:
            print(f"skipping {len(skipped)} {label}(s) ({flag} to keep): {', '.join(skipped)}")
        loaded = [(t, s) for t, s in loaded if not drop(s)]
    if not loaded:
        raise SystemExit("no sessions left to analyse (see the skipped files above)")
    trials = pd.concat([t for t, _ in loaded], ignore_index=True)
    sessions = pd.DataFrame([s for _, s in loaded])
    return trials, sessions
