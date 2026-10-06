"""Prolific bonus list from the experiment's data files.

Reads every catlearn_2x2x2_online_*.csv in a folder, takes each participant's coins from the
test trials (the trials whose coins are paid) plus any rounds credited when the test ended
early at its criterion (block_credit_coins on the block_summary row), recomputes the bonus with the same rule as
src/reward.js and checks it against what the experiment showed and saved. Writes:

  bonus_payments.txt   "PROLIFIC_PID,amount" per line: paste into Prolific's
                       "Bulk bonus payments" box (Submissions -> Bonus payments)
  bonus_report.csv     one row per file: IDs, coins, bonus, completed or not, checks

Only finished sessions (with a 'final' row) are paid by default; --include-incomplete adds
partial sessions with the coins they earned. Pilots (no PROLIFIC_PID) are skipped.
Keep the conversion arguments in sync with REWARD in src/config.js.

  python3 tools/bonus_payments.py ~/Downloads/catlearn_data
"""

import argparse
import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)   # the 'final' row holds the full interaction log


def bonus_dollars(coins, coins_per_dollar, min_bonus, max_bonus):
    usd = max(min_bonus, coins / coins_per_dollar)
    if max_bonus is not None:
        usd = min(max_bonus, usd)
    return round(usd + 1e-9, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="folder with the downloaded catlearn_2x2x2_online_*.csv files")
    parser.add_argument("--coins-per-dollar", type=float, default=500)
    parser.add_argument("--min-bonus", type=float, default=0.0)
    parser.add_argument("--max-bonus", type=float, default=None)
    parser.add_argument("--include-incomplete", action="store_true", help="also pay sessions without a final row")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.folder), "catlearn_2x2x2_online_*.csv")))
    if not files:
        sys.exit(f"no catlearn_2x2x2_online_*.csv files in {args.folder}")

    report, payments, seen = [], {}, set()
    for path in files:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        pid = next((r["prolific_pid"] for r in rows if r.get("prolific_pid")), "")
        paid = [r for r in rows if r.get("part") == "response" and r.get("coins_paid") == "true"]
        credit = sum(int(r["block_credit_coins"] or 0) for r in rows
                     if r.get("part") == "block_summary" and r.get("coins_paid") == "true" and r.get("block_credit_coins"))
        coins = sum(int(r["coins_delta"]) for r in paid) + credit
        final = next((r for r in rows if r.get("part") == "final"), None)
        usd = bonus_dollars(coins, args.coins_per_dollar, args.min_bonus, args.max_bonus)
        checks = []
        if final:
            if int(final["bonus_coins"]) != coins:
                checks.append(f"saved bonus_coins {final['bonus_coins']} != recomputed {coins}")
            if abs(float(final["bonus_usd"]) - usd) > 0.005:
                checks.append(f"saved bonus_usd {final['bonus_usd']} != recomputed {usd:.2f}")
        if pid and pid in seen:
            checks.append("duplicate PROLIFIC_PID (several files): paid once, check by hand")
        report.append({"file": os.path.basename(path), "prolific_pid": pid, "completed": bool(final),
                       "paid_trials": len(paid), "credit_coins": credit, "bonus_coins": coins, "bonus_usd": f"{usd:.2f}",
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
