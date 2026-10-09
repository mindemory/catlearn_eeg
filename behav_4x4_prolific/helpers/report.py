"""Text output: markdown tables."""

import numpy as np


def md_table(df):
    """A DataFrame as a markdown table (floats compact, NaN empty)."""
    cols = list(df.columns)
    fmt = lambda v: "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
