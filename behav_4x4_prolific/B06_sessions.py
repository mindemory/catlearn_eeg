"""Every session of the study and how it ended: finished, screened out (and why), or left
unfinished (a DataPipe .partial.json on Drive with no finished file).

Per session: Prolific ID, study, version, start time, the last screen reached, trials
answered, and the outcome (helpers.sessions.session_outcome):
  finished                     reached the final screen (bonus: what the task promised)
  screened out: <reason>       quiz (failed every attempt), practice_type_I (type-I practice
                               not passed within its trials), practice_timeouts, browser_check
  ended at the browser check   window too small (early sessions, before the browser check
                               used the screen-out path)
  unfinished                   only a partial file: returned, timed out, or still running
                               (sessions that a later file of the same participant finished
                               are marked "restarted, finished later")
  test run                     no Prolific ID (a run in the browser), or Prolific's
                               "Preview as participant" (study ID UNSAVED-STUDY)
For screen-outs at the type-I practice, why the gate stopped them: accuracy, the best run of
the criterion window (e.g. 9 of 12), P(F), late answers, median RT, and how often the choice
repeated the previous trial's correct answer. Exclusions of finished participants
(helpers.data.exclusions) are listed too; the reasons are in analysis/exclusions.csv.

Sessions of the F/J study only (task_version 1.x); files archived in params.ARCHIVES are left
out.

Outputs: analysis/sessions.csv and sessions.md

  ~/miniforge3/envs/kernelbehav/bin/python B06_sessions.py
"""

import pandas as pd

from helpers.report import md_table
from helpers.sessions import read_rows, session_outcome
from helpers.study import Study, parser
from params import ARCHIVES, DATA_DIR, DRIVE_DIR, SESSION_PATTERN, TASK_VERSION_PREFIX


def main():
    args = parser(__doc__).parse_args()
    study = Study.from_args(args)

    archived = {f.name for a in ARCHIVES if a.is_dir() for f in a.glob("*.csv")}
    rows = [session_outcome(read_rows(f), f.name, False) for f in sorted(DATA_DIR.glob(SESSION_PATTERN))]
    finished = {r["file"].removesuffix(".csv") for r in rows}
    for f in sorted(DRIVE_DIR.glob("catlearn_online_*.partial.json")) if DRIVE_DIR.is_dir() else []:
        key = f.name.removesuffix(".partial.json").rsplit("-", 1)[0]
        if key not in finished and f"{key}.csv" not in archived:
            rows.append(session_outcome(read_rows(f), f.name, True))
    df = pd.DataFrame(rows)
    df = df[df["task_version"].str.startswith(TASK_VERSION_PREFIX)].copy()
    done = set(df.loc[df["outcome"] == "finished", "prolific_pid"])
    restarted = (df["outcome"] == "unfinished") & df["prolific_pid"].isin(done)
    df.loc[restarted, "outcome"] = "unfinished (restarted, finished later)"
    df = df.sort_values("started_at")

    ex = study.exclusion_table.set_index("participant")
    df["excluded"] = df["prolific_pid"].map(lambda p: bool(ex.loc[p, "excluded"]) if p in ex.index else "")
    df["exclusion_rule"] = df["prolific_pid"].map(lambda p: ex.loc[p, "rule"] if p in ex.index else "")
    df["p_f_test"] = df["prolific_pid"].map(lambda p: round(ex.loc[p, "p_f"], 3) if p in ex.index else "")

    out = study.out()
    df.to_csv(out / "sessions.csv", index=False)
    show = df.assign(id=df["prolific_pid"].str[:8], start=df["started_at"].str[5:16].str.replace("T", " "))
    counts = show.groupby(["version", "outcome"]).size().rename("n").reset_index()
    lines = ["# Sessions", "", "## Outcomes", "", md_table(counts), "", "## Every session", "",
             md_table(show[["start", "id", "version", "outcome", "last_screen", "trials", "test_trials",
                            "excluded", "exclusion_rule"] + (["bonus_usd"] if "bonus_usd" in show else [])].fillna("")), ""]
    ti = show[show["outcome"] == "screened out: practice_type_I"]
    if len(ti):
        lines += ["## Screened out at the type-I practice", "",
                  md_table(ti[["id", "version", "trials", "type_i_acc", "type_i_best_window", "type_i_p_f",
                               "type_i_late", "type_i_median_rt", "type_i_repeat_prev_answer"]]), ""]
    (out / "sessions.md").write_text("\n".join(lines))
    print(counts.to_string(index=False))
    print(f"-> {out}/sessions.csv, sessions.md")


if __name__ == "__main__":
    main()
