# catlearn_2x2x2_prolific: slot-machine task, 2×2×2 (jsPsych)

Online category-learning experiment for Prolific, in jsPsych 8: the 2×2×2 version of
[`../catlearn_4x4_prolific`](../catlearn_4x4_prolific). The slot machine, calibration,
practice, timing and Prolific handling work as in that study. The differences:

- **Three symbols.** The test machines show three symbols.
- **Two test blocks.** Both are type III, version A+B+AC+BC, the same for everyone. Each
  ends early once the participant has learned the rule.
- **Noise patches.** The test symbols are 1/f^α noise patches, for now all at α = 2.00.
  Stimuli for α 1.50, 1.75 and 2.25 also exist, so α can become a between-participant
  condition (`STIMULI.noise.alphas`).

## The machine and a round

- **Layout.** Up to three symbols sit at the corners of an equilateral triangle around a
  bull's-eye fixation, each 6° from it, inside a slot-machine frame with a lever.
  - Positions: **A** bottom left, **B** bottom right, **C** top.
  - Each position shows one of 2 symbols.
  - Practice machines (2×2) use A and B only; the test machines (2×2×2) use all three.
- **A round:**

  | Step | What happens | Duration |
  |---|---|---|
  | wait | machine + fixation only; `SPACE` pulls the lever, or the round starts by itself | up to 3 s |
  | pull | the lever swings down and back | 0.4 s |
  | response | the symbols: "Will these symbols win?" `F` / `J` (either case) | up to 4 s |
  | feedback | green circles around every symbol if correct, red if wrong or late; the fixation stays on | 2 s |

  Which key means YES is randomised per participant (F = YES for about half).

## Design

`BLOCKS`, `RULES`, `TEST_CRITERION` and `STIMULI` in `src/config.js`. Every block is a new
machine with new symbols.

| Block | Machine | Symbols | Rule | Ends |
|---|---|---|---|---|
| Practice 1 | 2×2 (A, B) | fractals | type I, on A or B at random | 8 of the last 10 correct, or after 80 rounds |
| Practice 2 | 2×2 (A, B) | fractals | XOR | 8 of the last 10 correct, or after 80 rounds |
| Test 1 | 2×2×2 (A, B, C) | noise patches, one α | type III, A+B+AC+BC | **22 of the last 24 correct**, or after 152 rounds (19 passes of the 8 combinations) |
| Test 2 | 2×2×2 (A, B, C) | new noise patches, same α | the same rule | the same |

- **Test rule.** From `kernel_model/kernel_modes.py` (`Design('2x2x2')`), version `AB` of
  type III: kernel modes A + B + AC + BC.
  - The pairs that win (A, B, C levels): 000, 001, 010, 101.
- **Why 22 of 24 and not 75%.** In this rule any one or two features alone already give
  75% correct, so a 75% criterion would end the block for someone who only learned A. It
  would also end it for many guessers (simulated, 300 rounds):

  | Criterion | Guessing | Knows only A | Knows the rule, 85–95% correct |
  |---|---|---|---|
  | 12 of 16 (75%) | 91% stop | 100% | 100% |
  | 18 of 24 (75%) | 45% | 100% | 100% |
  | 21 of 24 (88%) | 1% | 30% | 100% |
  | **22 of 24 (92%)** | **0%** | **2%** | **100%** |

  So only participants who learned the interactions reach it.
- **Same rule, new machine.** Test 2 has the same rule with new patches. It shows whether
  the rule, once learned, is learned faster on new symbols.
- **Rounds left are paid.** When a test block ends at the criterion, its rounds left are
  paid as correct (`creditRemaining`). Learning fast therefore never costs bonus.
  - Saved as `block_credit_rounds` / `block_credit_coins` on the block's summary row.
- **Practice and test share nothing at the stimulus level.** Fractals and noise patches
  look nothing alike, and every machine is new. What practice does pass on is the idea that
  single symbols (type I) and combinations (XOR, on A and B) can decide the payout.

## Noise patches (test symbols)

- **What they are.** 1/f^α noise in a circular aperture with a soft edge, RMS-contrast
  matched (`task_design/noise_patches.py`), drawn 4° across on the grey background.
- **Which α.** `STIMULI.noise.alphas` in `src/config.js` lists the α values participants
  are assigned to, one each at random. **For now it is only 2.00.**
  - To make α a condition again, add `'1.50'`, `'1.75'` and/or `'2.25'`. Their stimuli are
    already made.
  - `?alpha=…` fixes the α for one participant, among those listed.
- **Equally distinct pairs.** Within an α, every pair is equally hard to tell apart: its
  DreamSim distance is within ±0.0013 of that α's median pair distance. Between α
  conditions, difficulty steps up with α:

  | α | DreamSim distance within a pair | Pairs |
  |---|---|---|
  | 1.50 | 0.058 | 30 |
  | 1.75 | 0.083 | 30 |
  | 2.00 | 0.108 | 30 |
  | 2.25 | 0.127 | 30 |
  | (fractal pairs, for comparison) | 0.256 | |

  Higher α means smoother, blobbier patches, which are easier to tell apart.
  `task_design/noise_similarity.py` shows how the distance depends on α.
- **How they were made.** `tools/make_noise_stimuli.py` makes the candidates (500 per α,
  in `~/Documents/data/catlearn_eeg/noise_pool/`), their DreamSim distances, and the pairs.
  - Output in `stimuli/noise/a<α>/`, with a contact sheet and report per α.
  - `stimuli/noise/groups.js` lists the pairs.
- **Per participant.** Each position of each test block shows one pair of the
  participant's α, six different pairs in all.

## Fractals (practice symbols)

- **Source.** The `2x2x2` set from [`../fractal_stimuli`](../fractal_stimuli): 48 pairs,
  each at a DreamSim distance within 0.0025 of 0.256. Every fractal has the same lightness
  profile, chroma and area.
- **Files.** `stimuli/fractal_groups/`.

## Coins and bonus

| Outcome | Coins |
|---|---|
| correct prediction | **+10** |
| wrong prediction | **−5** |
| late (no key within 4 s) | **−5** |

- **Coin bar.** A thin bar at the top of the screen shows the coins, filling toward the
  most the phase can pay. It is grey and labelled "practice" in practice blocks.
- **Bonus.** Test coins are paid on top of the Prolific base pay: **500 coins = $1**,
  never below $0.

  | Test outcome | Coins | Bonus |
  |---|---|---|
  | guessing all 2 × 152 rounds | ~760 | ~$1.52 |
  | 72% correct over 2 × 152 rounds | ~1,760 | ~$3.53 |
  | criterion reached in both (rest credited) | up to 3,040 | up to $6.08 |

## Run it locally (pilot)

The experiment is a static web page, so it needs nothing but a local web server. In a
terminal:

```bash
cd ~/Documents/codes/catlearn_eeg/catlearn_2x2x2_prolific
```

```bash
python3 -m http.server 8000
```

Then open the experiment in Chrome or Firefox:

| Address | What you get |
|---|---|
| <http://localhost:8000/> | **the real study**: full-length blocks (test up to 2 × 152 rounds, at most ~36 min). Use this for pilots whose data you want to analyse. |
| <http://localhost:8000/?debug=1&skip=intro,calibration> | a quick check that everything works (~3 min): shortened blocks, with a yellow "DEBUG RUN" banner at every block start |

Stop the server with Ctrl+C. After changing code, hard-reload the page (Cmd+Shift+R);
browsers cache the code.

At the end the data download as a CSV, `catlearn_2x2x2_online_pilot_<date-time>.csv`, to your
Downloads folder.

The page loads jsPsych from unpkg.com, so the pilot machine needs internet. Opening
`index.html` by double-clicking does not work; browsers block the code modules over
`file://`.

### URL options

| Option | Effect |
|---|---|
| `?debug=1` | **shortened blocks, for testing only**: every block at most 3 passes (practice 12 rounds, test 24 instead of 152). Every block start shows a DEBUG RUN banner, and the data have `debug = true` |
| `?skip=intro` | skip consent, instructions and the quiz |
| `?skip=calibration` | skip the card / blind-spot measurement (assumes 60 cm and a 96-dpi screen); combine as `?skip=intro,calibration` |
| `?seed=123` | fixed design (α, practice rule, symbols, round order, keys) |
| `?alpha=2.00` | fixed noise α, one of `STIMULI.noise.alphas` (now only `2.00`; `?alpha=2` works too) |
| `?simulate=1` | jsPsych plays the whole experiment by itself and saves the data (checks the data pipeline in seconds) |
| `?simulate=visual` | the same, but at real speed on screen |

Options combine, e.g. `?debug=1&skip=intro,calibration&seed=5`. With no `PROLIFIC_PID` in
the URL the run counts as a pilot (`participant = pilot`) and never redirects to Prolific.

## Sizes in degrees

Every position and size in `LAYOUT` (`src/config.js`) is in degrees of visual angle.
- **Noise patches:** 4° across (the aperture).
- **Fractals:** drawn at the EEG task's scale. 4° is the square that encloses the MATLAB
  task's fractals; the new fractals have the same median extent and area, and the spikiest
  reach 4.3°.
They are centred 6° from fixation at the corners of an equilateral triangle: A at
(−5.2°, −3°), B at (5.2°, −3°), C at (0°, 6°). The bull's eye is 0.6° (0.2° centre, 0.15° cross).

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
- **Window too small.** If the whole layout (24° × 21°, including the frame and lever)
  does not fill at most 85% of the window, everything shrinks by the same factor
  (`layout_scale` < 1). The triangle makes the layout tall: on a laptop screen at 60 cm,
  expect `layout_scale` around 0.8. The 85% margin keeps the top of the window free for the
  coin bar.
- **Why not webcam eye tracking?** It can't measure viewing distance reliably; the blind
  spot is the standard online method.

## Changing the design

Everything is in [`src/config.js`](src/config.js):

| Setting | What it holds |
|---|---|
| `RULES` | win (1) / lose (0) of every combination, for each 2×2 and 2×2×2 rule, with its levels per feature. Generated from the task definitions in `kernel_model/kernel_modes.py`, the same as the task-type figures; combination = 2a + b (2×2) or 4a + 2b + c (2×2×2). |
| `BLOCKS` | the blocks in order: phase, rule (a key or a list to pick from at random), symbol set, passes, and the criterion that ends the block early |
| `PRACTICE_CRITERION`, `TEST_CRITERION` | the window and number correct that end a block; for the test, whether the rounds left are then paid |
| `STIMULI` | the symbol sets (fractals, noise), their folders and drawing scale, and the α conditions |
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
| `src/design.js` | a participant's whole design from one seed: the α condition, rules, symbols, round order, which key means YES. Pure functions; the same seed always gives the same design |
| `src/calibration.js` | card + blind-spot calibration, pixels per degree, fit to the window |
| `src/reward.js` | coins per round and coins → dollars: the one place the payout rule is computed |
| `src/display.js` | HTML for every screen (machine and lever, symbols, bull's eye, feedback circles) in degrees, and the coin bar |
| `src/task.js` | blocks: intro → rounds (wait → pull → response → feedback) → block summary; block criteria and the credit for rounds left |
| `src/instructions.js` | consent, instructions, comprehension quiz (repeats until correct, up to 3 times) |
| `src/data.js` | Prolific IDs from the URL, tab-switch / full-screen tracking, saving |
| `src/main.js` | puts it together: browser check → consent → full screen → calibration → instructions + quiz → blocks → final screen → save → Prolific |
| `tools/bonus_payments.py` | Prolific bonus list from the data files, credited rounds included (see below) |
| `tools/make_noise_stimuli.py` | makes the noise-patch pairs for every α (see Noise patches) |
| `stimuli/fractal_groups/` | the practice fractals (the `2x2x2` fractal set), plus `groups.js` / `groups.json` |
| `stimuli/noise/` | the test noise patches, `a<α>/<n>.png`, and `groups.js` (the pairs per α) |

## Data

One CSV per participant, one row per screen. Analyse the rows with `part == "response"`
(one per round).

| Column | Meaning |
|---|---|
| `participant`, `prolific_pid`, `study_id`, `session_id` | Prolific IDs (`pilot` when run locally) |
| `seed`, `task_version` | regenerate the exact design with `buildDesign(seed, {alpha: noise_alpha})` |
| `noise_alpha`, `test_rule` | the participant's noise α (now always `2.00`) and the test rule (`x8_III_AB`) |
| `key_yes`, `key_no` | this participant's keys for YES (the symbols win) and NO |
| `block`, `phase`, `block_in_phase`, `rule`, `rule_type`, `levels`, `coins_paid` | block information (`levels`: `2x2` or `2x2x2`) |
| `trial_in_block`, `rep` | round number in the block, and which pass through the combinations |
| `compound`, `level_a`, `level_b`, `level_c` | which combination: `compound = 4a + 2b + c` (2×2: `2a + b`, `level_c` empty) |
| `stim_set`, `stim_dir` | the symbol set (`fractals` or `noise`) and its folder |
| `symbol_a`, `symbol_b`, `symbol_c` | file numbers in `stim_dir` shown bottom left (A), bottom right (B) and top (C); `symbol_c` empty in practice |
| `category`, `outcome`, `correct_key` | the right answer: 1 / `win` / `key_yes`, or 0 / `lose` / `key_no` |
| `response`, `choice`, `rt` | key pressed (lowercase), the answer (1 = YES), and RT in ms from symbol onset (`null` if late) |
| `correct`, `timeout` | outcome |
| `round_started_by`, `wait_ms` | `space` or `timeout` (started by itself after 3 s), and how long the wait screen lasted |
| `recent_correct` | correct answers among the last rounds of the block's criterion window (practice 10, test 24) |
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
| `block_summary` | `block_rounds`, `block_pcorrect`, `block_n_late`, `block_coins` (credit included), coins so far, `criterion_met`, and for the test `block_credit_rounds`, `block_credit_coins` |
| `final` | `test_pcorrect`, **`bonus_coins`, `bonus_usd`**, `practice_coins`, the complete tab-switch / full-screen log (`interaction_log`), `finished_at` |
| `wait`, `pull`, `feedback` | timing checks (`wait` also has `started_by`) |

Possible exclusion criteria to decide before data collection:

- `comprehension_passed == false`
- `criterion_met == false` in a practice block
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
   35 minutes (at most ~36 with every test round; less for participants who reach the
   criterion).
5. **Study description.** Say that participants need a **bank or ID card** for the screen
   setup, and mention the bonus ("up to ~$6 based on performance"); budget about $2–6 of bonus per participant.
6. **Consent.** Check that the text in `src/instructions.js` matches the IRB-approved
   version, including the performance bonus.
7. **Pilot first.** Check it yourself with `?debug=1`, then run the real study once in full, then on Prolific with
   a handful of participants, checking each CSV and a test bonus payment before launching
   the rest.

## Pilot: participants and cost

- **Who:** everyone at α = 2.00. The costs below are per person and per 10 people.
- **If α becomes a condition again:** for exact counts per α, run one Prolific study per
  α, with `&alpha=1.50` (etc.) added to its study URL. Exclude participants of the other
  studies.
- **Assumptions:** the same as the 4×4 deck's cost model.
  - Prolific: $12/h; 33.3% fee on all payments, bonuses included.
  - About 4.4 s per round, 40 rounds per practice block, 8 min of setup and instructions.

| | Per person | 10 people (with fee) |
|---|---|---|
| Session, at most (2 × 152 test rounds) | ~36 min | |
| Base pay (36 min at $12/h) | $7.20 | $96 |
| Bonus, everyone guessing | $1.52 | $20 |
| Bonus, typical (72% correct) | ~$3.53 | ~$47 |
| Bonus, everyone reaching the criterion | up to $6.08 | up to $81 |
| **Total** | | **~$116 (all guessing) · ~$143 typical · ~$177 at most** |

## Differences from the 4×4 study

- **Three positions.** The test machines show three symbols in an equilateral triangle
  (A bottom left, B bottom right, C top), each 6° from fixation, instead of two symbols
  6° left and right.
  - The machine frame is taller (19° × 16.5°, centred 1.5° above fixation).
  - The layout fills at most 85% of the window.
- **Test.** Two blocks of type III (A+B+AC+BC), new patches in the second, up to 152
  rounds each, each ending at 22 of the last 24 correct with the rounds left credited. The 4×4 study has three 128-round blocks of one
  4×4 type with an A → B switch.
- **Symbols.** Noise patches in the test (α = 2.00 for now); pairs of fractals in the
  practice.
- **Data.** Files are named `catlearn_2x2x2_online_*`.
  - Rounds have `symbol_a/b/c`, `stim_set` and `stim_dir` instead of `fractal_a/b`.
  - Rounds have `level_c`, and `levels` replaces `size`.
  - New columns: `noise_alpha`, `test_rule`.

The timing, keys, calibration, instructions and Prolific handling are the same; see the 4×4
README for the history of those choices.
