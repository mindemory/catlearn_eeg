"""Load the noise discrimination study's data files (noise_discrim_prolific) into tidy tables.

Each participant's CSV (one row per screen, see noise_discrim_prolific/README.md) becomes:
  trials    one row per trial (part == 'response')
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
from pathlib import Path

import pandas as pd

DATA_DIR = Path.home() / "Documents" / "data" / "catlearn_eeg" / "noise_discrim_prolific"
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
    return path.stem.replace("noise_discrim_", "")


def target_ms(df):
    """The intended exposure: TIMING.stimulus from the design row (version 1.0 had no record: 200 ms)"""
    design = df[df["part"] == "design"]
    if "timing_json" in df and len(design) and isinstance(design["timing_json"].iloc[0], str):
        return json.loads(design["timing_json"].iloc[0])["stimulus"]
    return 200


def load_file(path):
    df = pd.read_csv(path, low_memory=False)
    pid = participant_id(df, path)
    t = df[df["part"] == "response"][TRIAL_COLUMNS].copy()
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


def load_all(data_dir=DATA_DIR, include_debug=False):
    """Every data file in data_dir; debug runs (?debug=1, shortened) are left out unless include_debug"""
    files = sorted(Path(data_dir).glob("noise_discrim_*.csv"))
    if not files:
        raise SystemExit(f"no noise_discrim_*.csv files in {data_dir}")
    loaded = [load_file(f) for f in files]
    if not include_debug:
        skipped = [s["file"] for _, s in loaded if s["debug"]]
        if skipped:
            print(f"skipping {len(skipped)} debug run(s): {', '.join(skipped)} (--include-debug to keep)")
        loaded = [(t, s) for t, s in loaded if not s["debug"]]
        if not loaded:
            raise SystemExit("only debug runs found (--include-debug to analyse them)")
    trials = pd.concat([t for t, _ in loaded], ignore_index=True)
    sessions = pd.DataFrame([s for _, s in loaded])
    return trials, sessions
