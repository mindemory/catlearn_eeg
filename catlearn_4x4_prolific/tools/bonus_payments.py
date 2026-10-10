r"""Prolific bonus list from the experiment's data files.

Reads every catlearn_online_*.csv in a folder, takes each participant's answers in the
bonus blocks (the test blocks), recomputes the bonus with the same rule as src/bonus.js and
checks it against what the experiment showed and saved. Per bonus block, from its
proportion correct (late answers wrong), the amount of the highest tier reached (BONUS.tiers
in src/config.js):

  saved in each file as bonus_tiers:     below 56% $0 · 56% $0.25 · 60% $0.50 · 70% or more $1.00
  the first sessions (no bonus_tiers):  below 50% $0 · 50% $0.50 · 60% $1.00 · 70% or more $2.00
summed over blocks, rounded to cents. Each session is paid by the tiers it ran with.

Bonuses already paid are kept in a ledger, ROOT/bonus/paid.csv (prolific_pid, amount_usd,
paid_on, note), and left out of the list to pay. After paying the list on Prolific, run again
with --mark-paid to add it to the ledger.

Writes:
  bonus_payments.txt   "PROLIFIC_PID,amount" per line, the bonuses not paid yet
  bonus_payments_study_<STUDY_ID>.txt   the same per Prolific study: Prolific pays bonuses
                       per study, so paste each into that study's "Bulk bonus payments" box
                       (Submissions -> Bonus payments)
  bonus_report.csv     one row per file: IDs, rounds, proportion correct, bonus, paid, checks

Only finished sessions (with a 'final' row) are paid by default; --include-incomplete adds
partial sessions, paid for the blocks they finished. Screened-out sessions (screened_out
set; paid the fixed screen-out reward by Prolific) get no bonus. Test runs (no PROLIFIC_PID) and
files from the earlier slot-machine builds (task_version 3.x, paid in coins; recognised by
having no counts_for_bonus column) are skipped. Keep the arguments in sync with BONUS in
src/config.js (TIERS below).

By default it reads the finished sessions in ROOT/data/ (copied from Drive by
behav_4x4_prolific's analyses; see helpers/data.py there) and writes to ROOT/bonus/, where ROOT
is ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific:

  python3 tools/bonus_payments.py                  # the list to pay
  python3 tools/bonus_payments.py --mark-paid      # after paying it: record it in the ledger
  python3 tools/bonus_payments.py ~/My\ Drive/DataPipe/catlearn_4x4      # straight from Drive
"""

import argparse
import csv
import datetime
import glob
import json
import os
import sys

ROOT = os.path.expanduser("~/Documents/data/catlearn_eeg/catlearn_4x4_prolific")

csv.field_size_limit(sys.maxsize)   # the 'final' row holds the full interaction log


# BONUS.tiers in src/config.js. Every session saves the tiers it ran with (bonus_tiers, on the
# final row); the first sessions' files predate that and used FIRST_TIERS.
FIRST_TIERS = [(0.5, 0.5), (0.6, 1.0), (0.7, 2.0)]       # up to $2 per block
LEDGER_FIELDS = ["prolific_pid", "study_id", "amount_usd", "paid_on", "note"]


def tiers_for(rows):
    saved = next((r["bonus_tiers"] for r in rows if r.get("bonus_tiers")), "")
    return [tuple(t) for t in json.loads(saved)] if saved else FIRST_TIERS


def read_ledger(path):
    """Bonuses already paid: (prolific_pid, study_id) -> (amount, paid_on). Rows without a
    study_id (the first entries) match the participant in any study."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {(r["prolific_pid"], r.get("study_id") or ""): (float(r["amount_usd"]), r["paid_on"])
                for r in csv.DictReader(f)}


def paid_for(ledger, pid, study):
    return ledger.get((pid, study)) or ledger.get((pid, ""))


def block_bonus(p_correct, tiers):
    usd = 0.0
    for minimum, amount in tiers:
        if p_correct >= minimum:
            usd = amount
    return usd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default=os.path.join(ROOT, "data"),
                        help="folder with the catlearn_online_*.csv files (default: ROOT/data)")
    parser.add_argument("--out", default=os.path.join(ROOT, "bonus"),
                        help="folder for bonus_payments.txt and bonus_report.csv (default: ROOT/bonus)")
    parser.add_argument("--include-incomplete", action="store_true", help="also pay sessions without a final row")
    parser.add_argument("--ledger", default=os.path.join(ROOT, "bonus", "paid.csv"),
                        help="bonuses already paid (default: ROOT/bonus/paid.csv)")
    parser.add_argument("--mark-paid", action="store_true",
                        help="record the bonuses not paid yet as paid today (run after paying them on Prolific)")
    args = parser.parse_args()
    ledger = read_ledger(args.ledger)

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.folder), "catlearn_online_*.csv")))
    if not files:
        sys.exit(f"no catlearn_online_*.csv files in {args.folder}")

    report, payments, seen = [], {}, set()
    for path in files:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "counts_for_bonus" not in rows[0]:
            version = next((r["task_version"] for r in rows if r.get("task_version")), "?")
            why = (f"slot-machine build (task_version {version}), paid in coins" if version.startswith("3.")
                   else "no task trials (the session ended before the task, e.g. at the browser check)")
            report.append({"file": os.path.basename(path), "prolific_pid": "", "study_id": "", "completed": "", "bonus_rounds": "",
                           "p_correct": "", "block_p": "", "bonus_usd": "", "paid": "", "checks": f"skipped: {why}"})
            continue
        pid = next((r["prolific_pid"] for r in rows if r.get("prolific_pid")), "")
        study = next((r["study_id"] for r in rows if r.get("study_id")), "")
        counted = [r for r in rows if r.get("part") == "response" and r.get("counts_for_bonus") == "true"]
        # only blocks that reached their summary (finished) earn a bonus
        finished = {r["block"] for r in rows if r.get("part") == "block_summary" and r.get("counts_for_bonus") == "true"}
        blocks = {}
        for r in counted:
            if r["block"] in finished:
                blocks.setdefault(r["block"], []).append(r["correct"] == "true")
        p = sum(r["correct"] == "true" for r in counted) / len(counted) if counted else 0.0
        screened = next((r["screened_out"] for r in rows if r.get("screened_out")), "")
        tiers = tiers_for(rows)
        usd = 0.0 if screened else round(sum(block_bonus(sum(c) / len(c), tiers) for c in blocks.values()) + 1e-9, 2)
        final = next((r for r in rows if r.get("part") == "final"), None)
        checks = [f"screened out ({screened}): no bonus"] if screened else []
        if final and abs(float(final["bonus_usd"]) - usd) > 0.005:
            checks.append(f"saved bonus_usd {final['bonus_usd']} != recomputed {usd:.2f}")
        if pid and (pid, study) in seen:
            checks.append("duplicate PROLIFIC_PID in this study (several files): paid once, check by hand")
        elif pid and any(p == pid for p, _ in seen):
            checks.append("this participant also took another study: each study pays its own bonus")
        paid = paid_for(ledger, pid, study)
        if paid and abs(paid[0] - usd) > 0.005:
            checks.append(f"ledger says ${paid[0]:.2f} paid on {paid[1]}, bonus is ${usd:.2f}: check by hand")
        report.append({"file": os.path.basename(path), "prolific_pid": pid, "study_id": study, "completed": bool(final),
                       "bonus_rounds": len(counted), "p_correct": f"{p:.4f}",
                       "block_p": ";".join(f"{sum(c) / len(c):.3f}" for _, c in sorted(blocks.items(), key=lambda kv: int(kv[0]))),
                       "bonus_usd": f"{usd:.2f}", "paid": paid[1] if paid else "",
                       "checks": "; ".join(checks) or "ok"})
        if pid and (final or args.include_incomplete) and (pid, study) not in seen and usd > 0 and not paid:
            payments[(pid, study)] = usd
        if pid:
            seen.add((pid, study))

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "bonus_payments.txt"), "w") as f:
        f.writelines(f"{pid},{usd:.2f}\n" for (pid, _), usd in payments.items())
    # Prolific pays bonuses per study: one list per Prolific study as well
    for old in glob.glob(os.path.join(out_dir, "bonus_payments_study_*.txt")):
        os.remove(old)
    for study in sorted({s for _, s in payments}):
        mine = {pid: usd for (pid, s), usd in payments.items() if s == study}
        with open(os.path.join(out_dir, f"bonus_payments_study_{study}.txt"), "w") as f:
            f.writelines(f"{pid},{usd:.2f}\n" for pid, usd in mine.items())
        n, total = len(mine), sum(mine.values())
        print(f"  study {study}: {n} bonuses, ${total:.2f} -> bonus_payments_study_{study}.txt")
    with open(os.path.join(out_dir, "bonus_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)

    problems = [r for r in report if r["checks"] != "ok"]
    n_paid = sum(1 for r in report if r["paid"])
    print(f"{len(files)} files; {n_paid} bonuses already paid (ledger: {args.ledger}); "
          f"to pay now: {len(payments)}, total ${sum(payments.values()):.2f}")
    print(f"wrote bonus_payments.txt and bonus_report.csv to {out_dir}")
    if args.mark_paid and payments:
        new = not os.path.exists(args.ledger)
        with open(args.ledger, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            if new:
                w.writeheader()
            today = datetime.date.today().isoformat()
            w.writerows({"prolific_pid": pid, "study_id": study, "amount_usd": f"{usd:.2f}", "paid_on": today,
                         "note": "bonus_payments.txt of " + today} for (pid, study), usd in payments.items())
        print(f"recorded {len(payments)} bonuses as paid in {args.ledger}")
    for r in problems:
        print(f"  CHECK {r['file']}: {r['checks']}")


if __name__ == "__main__":
    main()
