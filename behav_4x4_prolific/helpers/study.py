"""The Study: one analysis run's data and where its outputs go.

    study = Study.from_args(args)          # loads (and by default syncs) every kept session
    study.save_exclusions()                # analysis/exclusions.csv, prints who is out
    for pid, version, r in study.participants():          # every participant with test trials
    groups = study.groups()                # {'A': [...], 'B': [...]}: kept, complete test blocks
    study.out("a_vs_b")                    # analysis/a_vs_b/, created

`kept` participants are those not excluded (helpers.data.exclusions); `complete` ones have as
many test trials as the most any participant of their version has (all three test blocks).
"""

import argparse

from helpers.data import exclusions, load_trials
from helpers.measures import test_blocks
from params import OUT_DIR, VERSIONS


def parser(doc):
    """An argument parser with the options every script shares (--no-sync)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-sync", action="store_true", help="don't copy new files from the Drive folder")
    return p


class Study:
    def __init__(self, sync=True, out_dir=OUT_DIR):
        self.trials = load_trials(sync=sync)
        self.out_dir = out_dir
        self.exclusion_table = exclusions(self.trials)
        self.excluded = set(self.exclusion_table.loc[self.exclusion_table["excluded"], "participant"])
        self._by_pid = dict(tuple(self.trials.groupby("participant", sort=False)))

    @classmethod
    def from_args(cls, args):
        return cls(sync=not args.no_sync)

    # ------------------------------------------------ exclusions
    def save_exclusions(self):
        """exclusions.csv in the analysis folder (every participant, excluded or not, and why);
        returns the set of excluded participants."""
        self.exclusion_table.to_csv(self.out() / "exclusions.csv", index=False)
        ex = self.exclusion_table[self.exclusion_table["excluded"]]
        if len(ex):
            print(f"excluded from averages ({len(ex)}): "
                  + ", ".join(f"{p[:8]} ({r})" for p, r in ex[["participant", "rule"]].values))
        return self.excluded

    def is_kept(self, pid):
        return pid not in self.excluded

    # ------------------------------------------------ participants
    def trials_of(self, pid):
        return self._by_pid[pid]

    def version(self, pid):
        return self._by_pid[pid]["version"].iloc[0]

    def test_trials(self, pid):
        return int((self._by_pid[pid]["phase"] == "test").sum())

    def test_blocks(self, pid):
        return test_blocks(self._by_pid[pid])

    def participants(self, version=None, kept=False):
        """(pid, version, trials) of every participant with test trials, in file order."""
        for pid, r in self._by_pid.items():
            v = r["version"].iloc[0]
            if self.test_trials(pid) == 0 or (version and v != version) or (kept and pid in self.excluded):
                continue
            yield pid, v, r

    def groups(self):
        """{version: [pid, ...]}: kept participants with complete test blocks."""
        out = {}
        for v in VERSIONS:
            pids = [p for p, _, _ in self.participants(version=v, kept=True)]
            full = max((self.test_trials(p) for p in pids), default=0)
            out[v] = [p for p in pids if self.test_trials(p) == full]
        return out

    @staticmethod
    def tag(groups):
        """'A: 10, B: 8' for figure titles."""
        return ", ".join(f"{v}: {len(groups.get(v, []))}" for v in VERSIONS)

    # ------------------------------------------------ outputs
    def out(self, *parts):
        """A folder in the analysis directory, created."""
        path = self.out_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
