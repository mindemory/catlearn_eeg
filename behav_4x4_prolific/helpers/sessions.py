"""What the raw session files say beyond the trials: session details, how each session ended,
and the questionnaire.

  session_info(df)       one finished session's details (device, calibration, minutes per
                         block and break, practice, test summaries, bonus)
  performance(trials, ex) test accuracy, binomial test against chance, P(F), exclusion
  demographics()         Prolific's demographic exports in params.PROLIFIC_DIR, or None
  read_rows(path)        a session file (CSV or DataPipe .partial.json) as a list of dicts
  session_outcome(rows)  finished / screened out (and why) / unfinished / test run
  questionnaire(path)    one participant's questionnaire answers, or None
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from helpers.measures import rule_type
from params import PROLIFIC_DIR

TRIAL_PARTS = ("iti", "response", "feedback")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def first(df, col):
    v = df[col].dropna() if col in df else pd.Series(dtype=object)
    return v.iloc[0] if len(v) else np.nan


def last(df, col):
    v = df[col].dropna() if col in df else pd.Series(dtype=object)
    return v.iloc[-1] if len(v) else np.nan


# ---------------------------------------------------------------- finished sessions
def session_info(df):
    """One participant's details from the raw session CSV. Minutes per block run from the end
    of its intro screen to its last trial; a break runs from the previous block's last trial
    to the end of this block's intro (summary, screen check, intro)."""
    df = df.reset_index(drop=True)
    t = num(df["time_elapsed"])
    minutes = lambda ms: ms / 60000   # noqa: E731
    pid = str(first(df, "prolific_pid"))
    out = {
        "participant": pid, "version": first(df, "version"), "task_version": first(df, "task_version"),
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

    prev_end = None
    test_k = practice_k = 0
    bonus = json.loads(last(df, "block_bonuses_usd")) if isinstance(last(df, "block_bonuses_usd"), str) else []
    for _, s in df[df["part"] == "block_summary"].iterrows():
        rows = df.index[df["part"].isin(TRIAL_PARTS) & (num(df["block"]) == num(s["block"]))]
        start, end = t[rows[0] - 1], t[rows[-1]]
        rule = df.loc[rows[0], "rule"]
        if s["phase"] == "practice":
            practice_k += 1
            key = f"practice{practice_k}"
            out[f"{key}_rule"] = rule_type(rule)
            out[f"{key}_trials"] = int(num(s["block_rounds"]))
            out[f"{key}_acc"] = round(num(s["block_pcorrect"]), 3)
            out[f"{key}_criterion_met"] = str(s["practice_criterion_met"]).lower() == "true"
        else:
            test_k += 1
            key = f"test{test_k}"
            out[f"{key}_rule"] = rule_type(rule)
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


def performance(trials, ex):
    """Test accuracy over all blocks, one-sided binomial test against chance, P(F) and
    exclusion (ex: helpers.data.exclusions, indexed by participant)."""
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
    """Prolific's demographic exports in params.PROLIFIC_DIR, one row per participant, or None."""
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


# ---------------------------------------------------------------- every session, finished or not
def read_rows(path):
    """A session file as a list of dicts: a finished CSV or a DataPipe .partial.json."""
    path = Path(path)
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []
    df = pd.read_csv(path, low_memory=False, dtype=str)
    return df.where(df.notna(), None).to_dict("records")


def first_value(rows, key):
    return next((r.get(key) for r in rows if r.get(key) not in (None, "", float("nan"))), "")


def session_outcome(rows, name, partial):
    """How a session ended: finished, screened out (and why), unfinished, or a test run (no
    Prolific ID, or Prolific's "Preview as participant", study ID UNSAVED-STUDY)."""
    parts = [r.get("part") for r in rows]
    resp = [r for r in rows if r.get("part") == "response"]
    pid = str(first_value(rows, "prolific_pid") or "")
    out = {"file": name, "prolific_pid": pid, "study_id": str(first_value(rows, "study_id") or ""),
           "task_version": str(first_value(rows, "task_version") or ""),
           "version": first_value(rows, "version"), "started_at": first_value(rows, "started_at"),
           "last_screen": parts[-1] if parts else "", "trials": len(resp),
           "test_trials": sum(r.get("phase") == "test" for r in resp), "partial": partial}
    screened = first_value(rows, "screened_out")
    if not pid or out["study_id"].startswith("UNSAVED"):
        out["outcome"] = "test run"
    elif "final" in parts:
        out["outcome"] = "finished"
        out["bonus_usd"] = first_value([r for r in rows if r.get("part") == "final"], "bonus_usd")
    elif screened:
        out["outcome"] = f"screened out: {screened}"
    elif out["last_screen"] == "browser_check":    # excluded by the browser check before it screened out
        out["outcome"] = "ended at the browser check"
    else:
        out["outcome"] = "unfinished"
    if screened == "practice_type_I":
        out.update(type_i_detail([r for r in resp if str(r.get("block")) in ("1", "1.0")]))
    return out


def type_i_detail(resp):
    """Why the type-I gate stopped a participant: accuracy, the best run of the criterion
    window, P(F), late answers, median RT, and how often the choice repeated the previous
    trial's correct answer (a strategy that ignores the symbols: with 2 F and 2 J pairs per
    pass it is right only about a third of the time)."""
    as_bool = lambda v: str(v).lower() == "true"   # noqa: E731
    correct = np.array([as_bool(r.get("correct")) for r in resp])
    choice = np.array([np.nan if r.get("choice") in (None, "") else float(r["choice"]) for r in resp])
    category = np.array([float(r.get("category")) for r in resp])
    rt = np.array([np.nan if r.get("rt") in (None, "") else float(r["rt"]) for r in resp])
    window = int(float(first_value(resp, "criterion_window") or 12))
    best = int(np.convolve(correct, np.ones(window, int), "valid").max()) if len(correct) >= window else int(correct.sum())
    answered = ~np.isnan(choice)
    prev = answered[1:]
    return {"type_i_acc": round(correct.mean(), 2), "type_i_best_window": f"{best} of {window}",
            "type_i_p_f": round(np.mean(choice[answered] == 1), 2),
            "type_i_late": round(1 - answered.mean(), 2), "type_i_median_rt": round(np.nanmedian(rt)),
            "type_i_repeat_prev_answer": round(np.mean(choice[1:][prev] == category[:-1][prev]), 2)}


# ---------------------------------------------------------------- questionnaire
def questionnaire(path):
    """One participant's questionnaire answers (q_* columns), minutes spent, and the Need for
    Cognition score (mean of 6 items on 1-5; items 3 and 4 reverse-scored); None without one."""
    # keep_default_na=False: answers such as "None" (video-game hours) are answers, not missing
    d = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])
    q = d[d["part"] == "questionnaire"]
    if q.empty:
        return None
    cols = [c for c in d.columns if c.startswith("q_")]
    row = {c: (q[c].dropna().iloc[0] if q[c].notna().any() else np.nan) for c in cols}
    row["participant"] = str(d["prolific_pid"].dropna().iloc[0])
    row["version"] = d["version"].dropna().iloc[0]
    row["questionnaire_min"] = pd.to_numeric(q["rt"], errors="coerce").sum() / 60000
    items = []
    for k in range(1, 7):
        rev = f"q_nfc{k}_rev"
        if rev in row and pd.notna(row[rev]):
            items.append(6 - float(row[rev]))
        elif f"q_nfc{k}" in row and pd.notna(row[f"q_nfc{k}"]):
            items.append(float(row[f"q_nfc{k}"]))
    row["nfc"] = np.mean(items) if items else np.nan
    return row
