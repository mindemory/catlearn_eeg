"""Parameters of the 4x4 study's analyses: paths, which sessions count, exclusion rules,
moving windows, statistics and colours. The scripts (B01-B10) and helpers/ read them from here.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path.home() / "Documents" / "data" / "catlearn_eeg" / "catlearn_4x4_prolific"
DATA_DIR = ROOT / "data"                         # every finished session (old_versions/: the slot-machine builds)
OUT_DIR = ROOT / "analysis"                      # figures and tables
DRIVE_DIR = Path.home() / "My Drive" / "DataPipe" / "catlearn_4x4"    # where DataPipe writes
PROLIFIC_DIR = ROOT / "prolific"                 # Prolific demographic exports, saved by hand
MANUAL_EXCLUSIONS = ROOT / "exclusions_manual.csv"   # participant, reason, decided_on (holds Prolific IDs,
                                                     # so it stays with the data, not in the repository)
FRACTAL_GROUPS_CSV = Path(__file__).resolve().parents[1] / "catlearn_4x4_prolific" / "stimuli" / "fractal_groups" / "groups.csv"
                                                 # task fractal file -> candidate in the fractal pool
FRACTAL_SIMILARITY_DIR = ROOT.parent / "fractal_pool_vivid" / "similarity"   # similarity_<measure>.npy,
                                                 # row k-1 = candidate k (fractal_stimuli/similarity_matrices.py)
SESSION_PATTERN = "catlearn_online_*.csv"        # finished sessions (not *.partial.json)
ARCHIVES = [DATA_DIR / "old_versions"]          # never copied back into DATA_DIR by the Drive sync

# ---------------------------------------------------------------- sessions
TASK_VERSION_PREFIX = "1."                       # the F/J study; 3.x were the earlier slot-machine builds
TRIAL_COLUMNS = ["participant", "version", "block", "phase", "rule", "trial_in_block", "rep", "compound",
                 "level_a", "level_b", "fractal_a", "fractal_b", "category", "choice", "correct", "timeout", "rt"]

# ---------------------------------------------------------------- design
TEST_TYPES = ["VI", "X", "II"]                   # the test blocks, in order, for both versions
MODES = ["A", "B", "AB"]                         # A = left symbol, B = right symbol, AB = their interaction
VERSIONS = ["A", "B"]

# ---------------------------------------------------------------- exclusions and checks
BIAS_LIMITS = (0.25, 0.75)                       # key bias: P(F) over the answered test trials outside this
FAST_RT_MS = 250                                 # an answer faster than this counts as fast
RUSH_FAST_SHARE = 0.20                           # rushing: in some test block, at least this share of the answers
                                                 # fast and that block at chance, AND the whole test at chance:
RUSH_CHANCE_ALPHA = 0.05                         # 'at chance' = accuracy not above 0.5 (one-sided binomial p >= this)

# ---------------------------------------------------------------- moving windows (trials, centred)
WINDOW = 16                                      # learning curves, mode heatmaps (one pass through the pairs)
EDGE = 48                                        # early vs late: the first and last EDGE trials of a block
LEVEL_WINDOW = 48                                # per-level and per-pair heatmaps (~12 trials per level)
CONTRAST_WINDOW = 32                             # version A vs B, modes and levels (~2 trials per pair)

# ---------------------------------------------------------------- stimulus features (B11)
FEATURES = ["colour", "shape", "perceptual"]     # colour histogram overlap, silhouette overlap, 1 - DreamSim
PAIR_KERNEL = "product"                          # pair similarity from the two symbols' similarities:
                                                 # "product" (both similar) or "sum" (either similar)

# ---------------------------------------------------------------- statistics
SEED = 0
N_PERM = 20000                                   # A vs B per mode
N_PERM_ROWS = 5000                               # A vs B per level and pair (many rows)
N_PERM_SANITY = 10000                            # A vs B sanity measures, questionnaire ratings

# ---------------------------------------------------------------- colours (dark background)
MEASURE_COLORS = {"accuracy": "#ffa62b", "f1": "#c77dff", "rt": "#4cc9f0"}   # mango, violet, sky blue
MEASURE_LABELS = {"accuracy": "Accuracy", "f1": "F1", "rt": "RT (ms)"}
VERSION_COLORS = {"A": "#ffa62b", "B": "#4cc9f0"}
BOX_COLOR = "#4cc9f0"                            # screening box plots
ASSOC_COLOR = "#ffa62b"                          # P(F) association box plots
EXCLUDED_COLOR = "#ff5a5f"                       # excluded participants, flagged values
MISSING_COLOR = "#3a3a3a"                        # heatmap cells with no value (e.g. a mode not in the rule)
DIFF_LIM = 0.3                                   # A - B heatmaps: colour scale +- this (accuracy units)
STRATEGY_COLORS = {"left": "#ffa62b", "right": "#4cc9f0", "separate": "#80ed99", "pairs": "#c77dff",
                   "guess": "#8d8d8d"}
