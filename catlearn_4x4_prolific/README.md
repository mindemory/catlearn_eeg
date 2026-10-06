# catlearn_4x4_prolific: slot-machine task (jsPsych)

Online category-learning experiment for Prolific, in jsPsych 8. It started as a rewrite of
Heeseung Lee's PsychoPy experiment (`~/Documents/heeseungStuff/Exp_CatLearn`). It is now a
self-paced slot machine with two symbols and a performance bonus.

- **The machine.** Two symbols (fractals) sit 6° left (A) and right (B) of a bull's-eye
  fixation, inside a slot-machine frame with a lever. Each side shows one of 2 fractals in
  practice (2×2) and one of 4 in the test (4×4).
- **A round:**

  | Step | What happens | Duration |
  |---|---|---|
  | wait | machine + fixation only; `SPACE` pulls the lever, or the round starts by itself | up to 3 s |
  | pull | the lever swings down and back | 0.4 s |
  | response | the two symbols: "Will this pair win?" `F` / `J` (either case) | up to 4 s |
  | feedback | green circles around both symbols if correct, red if wrong or late; the fixation stays on | 2 s |

  Which key means YES is randomised per participant (F = YES for about half).
- **Coins:**

  | Outcome | Coins |
  |---|---|
  | correct prediction | **+10** |
  | wrong prediction | **−5** |
  | late (no key within 4 s) | **−5** |

  A thin bar at the top of the screen shows the coins (labelled in coins), filling toward
  the most the phase can pay. It is grey and labelled "practice" in practice blocks.
- **The bonus.** Coins from the test blocks are paid on top of the Prolific base pay:
  **500 coins = $1**, never below $0. Participants see coins throughout; the dollar
  amount appears only in the instructions (the rate) and on the final screen.

  | Performance on the 384 test rounds | Coins | Bonus |
  |---|---|---|
  | guessing | ~960 | ~$1.92 |
  | 80% correct | ~2,690 | ~$5.38 |
  | perfect | 3,840 | $7.68 |

- **The design** (`BLOCKS`, `TEST_TYPES` in `src/config.js`); every block is a new machine
  with new fractals:
  1. **Practice 1** (2×2): type I, on A or B at random.
  2. **Practice 2** (2×2): XOR.
  3. **Test** (4×4): 3 blocks of 128 rounds (8 passes of the 16 pairs). Each participant
     gets one type at random from `STUDY` in `src/config.js`:
     - **Pilot** (`STUDY.phase = 'pilot'`): types VI, X, XV and XVI, 10 participants each.
     - **Main study** (`'main'`): all 7 types, I, VI, X, XII, XIV, XV and XVI, 30 each.
     - **Exact counts:** random draws don't give exact counts per type. For those, run one
       Prolific study per type, with `&type=XV` (etc.) added to its study URL and that many
       places, and exclude participants of the other studies. `&first=A` / `&first=B`
       also fixes the version order, for exactly half each.
     - Blocks 1–2 use its A-dominant or its B-dominant version (random). Block 3 uses the
       other version (A and B swapped).
     - Type XVI is pure XOR and has no A/B version, so its third block only brings new
       fractals.

  Each practice block ends once 8 of the last 10 answers are correct
  (`PRACTICE_CRITERION`), or after 80 rounds at the latest (flagged in the data).

## Fractals

The symbols are made so that no fractal stands out and every block is equally hard to tell
apart. They come from [`../fractal_stimuli`](../fractal_stimuli), which documents how they
are made:

- **Matched:** every fractal has the same lightness profile, chroma and area.
- **Groups:** the set has 18 groups of 4 (test) and 12 pairs (practice). Within each
  group, every pairwise DreamSim distance (a perceptual similarity network) is within
  0.0075 of 0.256, so every group is equally hard.
- **No near-duplicates:** no two of the 96 fractals are closer than 0.162.

Each side of a block shows one whole group, which is never used again for that
participant. `stimuli/fractal_groups/` also holds the groups (`groups.js`, imported by
`src/design.js`; `groups.json`; `groups.csv`), a contact sheet and a distance report.

## Run it locally (pilot)

The experiment is a static web page, so it needs nothing but a local web server. In a
terminal:

```bash
cd ~/Documents/codes/catlearn_eeg/catlearn_4x4_prolific
```

```bash
python3 -m http.server 8000
```

Then open the experiment in Chrome or Firefox:

| Address | What you get |
|---|---|
| <http://localhost:8000/> | **the real study**: full-length blocks (3 test blocks × 128 rounds, ~40 min). Use this for pilots whose data you want to analyse. |
| <http://localhost:8000/?debug=1&skip=intro,calibration> | a quick check that everything works (~3 min): shortened blocks, with a yellow "DEBUG RUN" banner at every block start |

Stop the server with Ctrl+C. After changing code, hard-reload the page (Cmd+Shift+R);
browsers cache the code.

At the end the data download as a CSV, `catlearn_online_pilot_<date-time>.csv`, to your
Downloads folder.

The page loads jsPsych from unpkg.com, so the pilot machine needs internet. Opening
`index.html` by double-clicking does not work; browsers block the code modules over
`file://`.

### URL options

| Option | Effect |
|---|---|
| `?debug=1` | **shortened blocks, for testing only**: test blocks 1 pass (16 rounds instead of 128), practice blocks at most 12 rounds. Every block start shows a DEBUG RUN banner, and the data have `debug = true` |
| `?skip=intro` | skip consent, instructions and the quiz |
| `?skip=calibration` | skip the card / blind-spot measurement (assumes 60 cm and a 96-dpi screen); combine as `?skip=intro,calibration` |
| `?seed=123` | fixed design (test type and order, rules, fractals, round order, keys) |
| `?type=XV` | fixed test type (any key of `TEST_TYPES`, also outside the current `STUDY` phase) |
| `?first=A` | fixed version order: A-dominant version in blocks 1–2 (`B` for the reverse) |
| `?simulate=1` | jsPsych plays the whole experiment by itself and saves the data (checks the data pipeline in seconds) |
| `?simulate=visual` | the same, but at real speed on screen |

Options combine, e.g. `?debug=1&skip=intro,calibration&seed=5`. With no `PROLIFIC_PID` in
the URL the run counts as a pilot (`participant = pilot`) and never redirects to Prolific.

## Sizes in degrees

Every position and size in `LAYOUT` (`src/config.js`) is in degrees of visual angle. The
symbols are drawn at the EEG task's scale: 4° is the square that encloses the MATLAB task's
fractals. The new fractals have the same median extent and area; the spikiest reach 4.3°.
They are centred 6° left and right of fixation. The bull's eye is 0.6° (0.2° centre, 0.15° cross).

To turn degrees into pixels the experiment needs each participant's screen and distance.
After full screen, `src/calibration.js` runs jsPsych's virtual chinrest (~2 min):

1. **Card step.** The participant resizes an on-screen card to match a real credit/ID card
   held against the screen, which gives pixels per mm.
2. **Blind-spot step.** With the right eye covered, they look straight at a black square and
   press space when a moving red ball vanishes into their blind spot, which gives the
   viewing distance. The square and ball are enlarged here; the plugin measures their
   rendered centres, so this does not bias the estimate.

From these, pixels per degree = 2 · distance · tan(0.5°) · pixels per mm, applied as the
CSS variable `--deg`.

- **Implausible measurements.** If the distance is outside 30–100 cm or the card outside
  1.5–12 px/mm, the measurement is repeated once. If it is still off, the experiment
  assumes 60 cm and/or a 96-dpi screen, and records which in `calibration_source`.
- **Window too small.** If the whole layout (24° × 12.5°, including the frame and lever)
  does not fit the window, everything shrinks by the same factor (`layout_scale` < 1).
- **Why not webcam eye tracking?** It can't measure viewing distance reliably; the blind
  spot is the standard online method.

## Changing the design

Everything is in [`src/config.js`](src/config.js):

| Setting | What it holds |
|---|---|
| `RULES` | win (1) / lose (0) of every pair, for each 2×2 and 4×4 rule. Generated from the task definitions in `kernel_model/kernel_modes.py`, the same as the task-type figures; pair = size · a + b. |
| `TEST_TYPES` | the test types, each as [A-dominant, B-dominant] rule |
| `BLOCKS` | the blocks in order: phase, rule (a key, a list to pick from at random, or `test:first` / `test:second`), passes, and whether a practice criterion ends the block |
| `PRACTICE_CRITERION` | the window and number correct that end a practice block |
| `REWARD` | coins per correct / wrong / late answer, which phases are paid, coins per dollar, minimum and maximum bonus |
| `TIMING`, `KEYS`, `LAYOUT`, `FEEDBACK` | durations (ms), keys, positions and sizes in degrees, feedback colours |
| `CALIBRATION` | blind-spot repetitions, plausible ranges, the values assumed when a measurement fails, fit margin |
| `BROWSER` | minimum window size; phones and tablets are excluded |
| `PROLIFIC` | completion codes |
| `DATA` | where data go |

The instructions compute their examples (coins, session length) from these settings, so
they stay correct when the design changes.

## Files

| File | What it does |
|---|---|
| `index.html` | loads jsPsych 8.3.0 and plugins (pinned versions), then `src/main.js` |
| `src/config.js` | all settings (above) |
| `src/design.js` | a participant's whole design from one seed: test type and version order, rules, fractals, round order, which key means YES. Pure functions; the same seed always gives the same design |
| `src/calibration.js` | card + blind-spot calibration, pixels per degree, fit to the window |
| `src/reward.js` | coins per round and coins → dollars: the one place the payout rule is computed |
| `src/display.js` | HTML for every screen (machine and lever, symbols, bull's eye, feedback circles) in degrees, and the coin bar |
| `src/task.js` | blocks: intro → rounds (wait → pull → response → feedback) → block summary; practice criterion |
| `src/instructions.js` | consent, instructions, comprehension quiz (repeats until correct, up to 3 times) |
| `src/data.js` | Prolific IDs from the URL, tab-switch / full-screen tracking, saving |
| `src/main.js` | puts it together: browser check → consent → full screen → calibration → instructions + quiz → blocks → final screen → save → Prolific |
| `tools/bonus_payments.py` | Prolific bonus list from the data files (see below) |
| `stimuli/fractal_groups/` | the 96 fractals the task uses, plus `groups.js` / `groups.json` (which fractals form each group) |

## Data

One CSV per participant, one row per screen. Analyse the rows with `part == "response"`
(one per round).

| Column | Meaning |
|---|---|
| `participant`, `prolific_pid`, `study_id`, `session_id` | Prolific IDs (`pilot` when run locally) |
| `seed`, `task_version` | regenerate the exact design with `buildDesign(seed, {type: test_type, first: test_first})` |
| `study_phase` | `pilot` or `main` (`STUDY.phase` when the data were collected) |
| `test_type`, `test_first` | the participant's test type (I … XVI) and which version came first (A or B) |
| `key_yes`, `key_no` | this participant's keys for YES (pair wins) and NO |
| `block`, `phase`, `block_in_phase`, `rule`, `rule_type`, `size`, `coins_paid` | block information |
| `trial_in_block`, `rep` | round number in the block, and which pass through the pairs |
| `compound`, `level_a`, `level_b` | which pair: `compound = size · a + b` |
| `fractal_a`, `fractal_b` | fractal file numbers shown left (A) and right (B) |
| `category`, `outcome`, `correct_key` | the right answer: 1 / `win` / `key_yes`, or 0 / `lose` / `key_no` |
| `response`, `choice`, `rt` | key pressed (lowercase), the answer (1 = YES), and RT in ms from symbol onset (`null` if late) |
| `correct`, `timeout` | outcome |
| `round_started_by`, `wait_ms` | `space` or `timeout` (started by itself after 3 s), and how long the wait screen lasted |
| `recent_correct` | correct answers among the last 10 rounds (the practice criterion) |
| `coins_delta`, `coins_block` | coins for this round, and the block total so far |
| `coins_bonus_total`, `coins_practice_total` | paid and practice coin totals so far |
| `blur_count`, `fullscreen_exit_count` | how often the participant left the tab/window or full screen so far (attention checks) |
| `px_per_deg`, `layout_scale`, `window_width`, `window_height` | screen scale on this round (degrees actually shown = LAYOUT value × `layout_scale`) |
| `time_elapsed` | ms since the start |

Other rows:

| `part` | What it records |
|---|---|
| `browser_check` | window size, browser, OS, mobile, refresh rate (`vsync_rate`, and `low_refresh` if under 50 Hz) |
| `consent` | consent given or not |
| `calibration` | each card + blind-spot attempt (`px2mm`, `view_dist_mm`, `calibration_plausible`) |
| `calibration_summary` | the values used: `calibration_source` (measured / assumed_distance / assumed_card / assumed_both / skipped), `px_per_mm`, `view_dist_mm`, `px_per_deg`, `layout_scale` |
| `comprehension` | quiz answers, `comprehension_passed`, `comprehension_attempt` |
| `design` | the full design as JSON: rules, fractals |
| `block_summary` | `block_rounds`, `block_pcorrect`, `block_n_late`, `block_coins`, coins so far, and for practice `practice_criterion_met` |
| `final` | `test_pcorrect`, **`bonus_coins`, `bonus_usd`**, `practice_coins`, the complete tab-switch / full-screen log (`interaction_log`), `finished_at` |
| `wait`, `pull`, `feedback` | timing checks (`wait` also has `started_by`) |

Possible exclusion criteria to decide before data collection:

- `comprehension_passed == false`
- `practice_criterion_met == false`
- many timeouts, or most rounds `started_by == timeout`
- high `blur_count`
- `low_refresh`
- `calibration_source` other than `measured`, or a small `layout_scale`

## Paying the bonus

Prolific pays bonuses separately from the base pay, after you approve submissions.

1. Put all participants' CSV files in one folder.
2. Run the payment script:

   ```bash
   python3 tools/bonus_payments.py ~/path/to/data_folder
   ```

   It recomputes every bonus from the paid rounds, checks it against the `bonus_usd` the
   experiment showed and saved, and writes two files:
   - `bonus_payments.txt` (`PROLIFIC_PID,amount` per line)
   - `bonus_report.csv` (every file with its checks)

   Pilots (no Prolific ID) are skipped. Duplicate IDs are paid once and flagged.
   Unfinished sessions are left out unless you pass `--include-incomplete`.
3. On Prolific, open the study → Submissions → **Bulk bonus payments**, and paste the
   contents of `bonus_payments.txt`.

If you change `REWARD` in `src/config.js`, pass the same conversion to the script
(`--coins-per-dollar`, `--min-bonus`, `--max-bonus`).

## Going live on Prolific: checklist

1. **Hosting and data.** `DATA.save = 'local'` only downloads to the participant's
   computer, which is useless online. Pick one backend and add it in
   [`src/data.js`](src/data.js) `saveData()`:

   | Backend | How it works | Cost |
   |---|---|---|
   | **DataPipe** | page on GitHub Pages, data to your OSF project | free, ~10 lines |
   | **JATOS** | your own server or MindProbe | free |
   | **Pavlovia** | hosts and stores | paid per participant unless Dartmouth has a site licence |

2. **Prolific study URL.** Your hosted address followed by
   `?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`.
   Prolific fills in the IDs.
3. **Completion code.** Copy it from Prolific into `PROLIFIC.completionCode`; participants
   are redirected with it automatically after saving.
4. **Device.** Restrict to desktop (laptop / computer), and set the estimated time: about
   40–45 minutes with the default design.
5. **Study description.** Say that participants need a **bank or ID card** for the screen
   setup, and mention the bonus ("up to ~$7.50 based on performance"); budget about $4–6 of bonus per participant.
6. **Consent.** Check that the text in `src/instructions.js` matches the IRB-approved
   version, including the performance bonus.
7. **Pilot first.** Check it yourself with `?debug=1`, then run the real study once in full, then on Prolific with
   a handful of participants, checking each CSV and a test bonus payment before launching
   the rest.

## Changes from the PsychoPy version

- **Slot machines.** The Friend-or-Enemy flags became slot machines: win/lose predictions,
  coins (+10 correct, −5 wrong or late) and a test-block bonus.
- **Two symbols.** Two symbols (left / right) instead of three in a triangle. A 2×2 practice
  ends on a performance criterion, followed by a 4×4 test with one type per participant
  and an A → B (or B → A) switch in block 3.
- **Self-paced rounds.** Each round starts on `SPACE` (a lever pull), or by itself after 3 s.
- **Sizes in degrees.** Instead of fractions of the screen height: 4° symbols at 6°, as in
  the EEG task, with a per-participant screen calibration.
- **Feedback without text.** Green or red circles around the symbols, with the fixation
  still on, replace PsychoPy's text over the symbols. A coin bar at the top of the screen
  replaces the trial counter and running score.
- **Fixation.** The MATLAB task's bull's eye replaces the cross and is shown throughout; the
  PsychoPy file had its cross disabled, although the instructions ask participants to look
  at it.
- **Keys.** `F` / `J` (either case), with YES/NO randomised per participant; PsychoPy used
  `E` / `F` with a fixed mapping.
- **Text instead of slides.** Instructions are HTML text rather than Keynote slides, and a
  comprehension quiz was added. The PsychoPy file also listed two instruction slides that
  don't exist (`instruction.008/.009`).
- **Eye tracking.** The calibration and fixation-check routines were dropped (lab only).
- **Logging.** Added Prolific IDs, the design seed, a browser check, full screen, tab-switch
  and full-screen-exit counts, and save/redirect handling.
