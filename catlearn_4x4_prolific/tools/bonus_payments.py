"""Prolific bonus list from the experiment's data files.

Reads every catlearn_online_*.csv in a folder, takes each participant's answers in the
bonus blocks (the test blocks), recomputes the bonus with the same rule as src/bonus.js and
checks it against what the experiment showed and saved. Per bonus block, from its
proportion correct (late answers wrong), the amount of the highest tier reached (BONUS.tiers
in src/config.js):

  below 50% $0 · 50% $0.50 · 60% $1.00 · 70% or more $2.00; summed over blocks, rounded to cents

Writes:
  bonus_payments.txt   "PROLIFIC_PID,amount" per line: paste into Prolific's
                       "Bulk bonus payments" box (Submissions -> Bonus payments)
  bonus_report.csv     one row per file: IDs, rounds, proportion correct, bonus, checks

Only finished sessions (with a 'final' row) are paid by default; --include-incomplete adds
partial sessions, paid for the blocks they finished. Screened-out sessions (screened_out
set; paid the fixed screen-out reward by Prolific) get no bonus. Pilots (no PROLIFIC_PID) and files
from the earlier slot-machine pilot builds (task_version 3.x, paid in coins; recognised by
having no counts_for_bonus column) are skipped. Keep the arguments in sync with BONUS in
src/config.js (TIERS below).

  python3 tools/bonus_payments.py ~/Downloads/catlearn_data
"""

import argparse
import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)   # the 'final' row holds the full interaction log


TIERS = [(0.5, 0.5), (0.6, 1.0), (0.7, 2.0)]   # BONUS.tiers in src/config.js


def block_bonus(p_correct):
    usd = 0.0
    for minimum, amount in TIERS:
        if p_correct >= minimum:
            usd = amount
    return usd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="folder with the downloaded catlearn_online_*.csv files")
    parser.add_argument("--include-incomplete", action="store_true", help="also pay sessions without a final row")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.folder), "catlearn_online_*.csv")))
    if not files:
        sys.exit(f"no catlearn_online_*.csv files in {args.folder}")

    report, payments, seen = [], {}, set()
    for path in files:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "counts_for_bonus" not in rows[0]:
            version = next((r["task_version"] for r in rows if r.get("task_version")), "?")
            report.append({"file": os.path.basename(path), "prolific_pid": "", "completed": "", "bonus_rounds": "",
                           "p_correct": "", "block_p": "", "bonus_usd": "",
                           "checks": f"skipped: slot-machine pilot build (task_version {version}), paid in coins"})
            continue
        pid = next((r["prolific_pid"] for r in rows if r.get("prolific_pid")), "")
        counted = [r for r in rows if r.get("part") == "response" and r.get("counts_for_bonus") == "true"]
        # only blocks that reached their summary (finished) earn a bonus
        finished = {r["block"] for r in rows if r.get("part") == "block_summary" and r.get("counts_for_bonus") == "true"}
        blocks = {}
        for r in counted:
            if r["block"] in finished:
                blocks.setdefault(r["block"], []).append(r["correct"] == "true")
        p = sum(r["correct"] == "true" for r in counted) / len(counted) if counted else 0.0
        screened = next((r["screened_out"] for r in rows if r.get("screened_out")), "")
        usd = 0.0 if screened else round(sum(block_bonus(sum(c) / len(c)) for c in blocks.values()) + 1e-9, 2)
        final = next((r for r in rows if r.get("part") == "final"), None)
        checks = [f"screened out ({screened}): no bonus"] if screened else []
        if final and abs(float(final["bonus_usd"]) - usd) > 0.005:
            checks.append(f"saved bonus_usd {final['bonus_usd']} != recomputed {usd:.2f}")
        if pid and pid in seen:
            checks.append("duplicate PROLIFIC_PID (several files): paid once, check by hand")
        report.append({"file": os.path.basename(path), "prolific_pid": pid, "completed": bool(final),
                       "bonus_rounds": len(counted), "p_correct": f"{p:.4f}",
                       "block_p": ";".join(f"{sum(c) / len(c):.3f}" for _, c in sorted(blocks.items(), key=lambda kv: int(kv[0]))),
                       "bonus_usd": f"{usd:.2f}",
                       "checks": "; ".join(checks) or "ok"})
        if pid and (final or args.include_incomplete) and pid not in seen and usd > 0:
            payments[pid] = usd
        if pid:
            seen.add(pid)

    out_dir = os.path.expanduser(args.folder)
    with open(os.path.join(out_dir, "bonus_payments.txt"), "w") as f:
        f.writelines(f"{pid},{usd:.2f}\n" for pid, usd in payments.items())
    with open(os.path.join(out_dir, "bonus_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)

    problems = [r for r in report if r["checks"] != "ok"]
    print(f"{len(files)} files, {len(payments)} bonuses, total ${sum(payments.values()):.2f}")
    print(f"wrote bonus_payments.txt and bonus_report.csv to {out_dir}")
    for r in problems:
        print(f"  CHECK {r['file']}: {r['checks']}")


if __name__ == "__main__":
    main()
