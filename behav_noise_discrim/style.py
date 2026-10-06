"""Dark plot theme for these analyses: behav_analyses/plot_style.py, loaded under its own
module name. (task_design/ has a different plot_style.py, which kernel_model/kernel_modes.py
imports; loading both as "plot_style" would make one shadow the other.)"""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "behav_analyses" / "plot_style.py"
_spec = importlib.util.spec_from_file_location("behav_plot_style", _path)
_theme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_theme)

ACCENT = _theme.ACCENT
CORRECT = _theme.CORRECT
FOREGROUND = _theme.FOREGROUND
INCORRECT = _theme.INCORRECT
apply_dark_theme = _theme.apply_dark_theme
