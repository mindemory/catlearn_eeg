"""Load 2x2 task behavior from the MATLAB output files into a trial-level DataFrame."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

DATA_DIR = Path("/Users/mrugank/Documents/data/catlearn_eeg/rl_generalization")


def load_block(path):
    """One row per trial of a single block file."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["matFile"]
    params, resp = mat.parameters, np.atleast_1d(mat.respReport)
    rows = [
        {
            "compound": r.compound,
            "levelA": r.levelA,
            "levelB": r.levelB,
            "category": r.category,
            "key": r.key if isinstance(r.key, str) else "",
            "correct_key": r.correctKey,
            "correct": bool(r.correct),
            "timed_out": bool(r.timedOut),
            "rt": np.nan if r.timedOut else float(r.RT),
        }
        for r in resp
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "trial", np.arange(1, len(df) + 1))
    df.insert(0, "block", int(params.block))
    df.insert(0, "config", mat.stim.configName)
    df.insert(0, "task", mat.subjectInfo.taskName)
    df.insert(0, "subject", params.subject)
    return df


def load_subject(subject):
    """All blocks of one subject, with a running trial count across blocks."""
    files = sorted((DATA_DIR / f"sub{subject}").glob("block*/matFile_task2by2_*.mat"))
    if not files:
        raise FileNotFoundError(f"no 2x2 block files for sub{subject} in {DATA_DIR}")
    df = pd.concat([load_block(f) for f in files], ignore_index=True)
    df.insert(df.columns.get_loc("trial") + 1, "trial_overall", np.arange(1, len(df) + 1))
    return df
