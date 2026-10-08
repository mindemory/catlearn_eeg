# catlearn_4x4_prolific: F/J category learning, 4×4 (jsPsych)

Online category-learning experiment for Prolific, in jsPsych 8. It started as a rewrite of
Heeseung Lee's PsychoPy experiment (`~/Documents/heeseungStuff/Exp_CatLearn`), was a slot
machine with coins in the pilot builds (task_version 3.x), and is now (task_version 1.0, the
first study) a plain F/J categorisation task with a performance bonus.

- **The screen.** Two symbols (fractals) sit 6° left (A) and right (B) of the centre.
  Participants look freely while the symbols are on; a bull's-eye fixation appears only
  between rounds, so every round starts from the centre. Each side shows one of 2
  fractals in practice (2×2) and one of 4 in the test
  (4×4).
- **The task.** Every pair belongs to one of two categories. Participants press **F** for
  one and **J** for the other; the keys are the same for everyone (`KEYS`: F = category 1
  of `RULES`). They learn from the feedback after every answer.
- **A round:**

  | Step | What happens | Duration |
  |---|---|---|
  | iti | fixation only (participants are asked to look at it) | 1 s |
  | response | the two symbols, no fixation (free viewing), until `F` / `J` (either case) | up to 4 s |
  | feedback | green circles around both symbols if correct, red if wrong or too slow; no text | 2 s |

  Rounds follow each other automatically; there is a break screen between blocks, which
  also asks participants to sit at about the same distance from the screen as at the start
  (the screen calibration is done once, at the start).
- **No points on screen.** There are no coins, scores or bars during the rounds.
  Participants are told about the bonus in the instructions, see each test block's bonus on
  its break screen, and the total on the final screen.
- **The bonus** (`BONUS` in `src/config.js`). It is paid on top of the Prolific base pay,
  per test block, from that block's proportion correct; late answers count as wrong:

  | Correct in the block | Bonus for the block |
  |---|---|
  | below 50% | $0.00 |
  | 50% to 60% | $0.50 |
  | 60% to 70% | $1.00 |
  | 70% or more | $2.00 |

  Up to $6 over the three test blocks. There are no penalties, so the bonus never reduces
  the base pay. 50% is chance: a pure guesser reaches 50% in about half the blocks and
  averages about $0.80 in all.
- **Screening** (`SCREENING` in `src/config.js`, Prolific "custom screening"). Participants
  who are not a good fit leave early with a fixed $1.00 and don't use up a place:

  | Check | When |
  |---|---|
  | the comprehension quiz failed twice (`COMPREHENSION.maxAttempts`) | right after the instructions, ~5 min in |
  | the type-I practice not passed (8 of the last 10 correct within 80 rounds) | after practice 1 |
  | more than 20% of practice rounds unanswered | after each practice block |

  - **Following Prolific's policy:** comprehension checks come right after the
    instructions, with two attempts. The screens are early in the session, and screen-outs
    get a fixed payment.
  - **Not a screen:** the XOR practice. It is genuinely hard, and screening on it would
    keep only good learners.
  - **What the participant sees:** a thank-you page, then a return to Prolific with the
    study's screen-out code (`PROLIFIC.screenOutCodes`).
  - **Data:** saved as usual, with `screened_out` = `quiz`, `practice_type_I` or
    `practice_timeouts` on every row.
  - **Test runs:** debug and simulated runs skip screening unless the URL has `?screen=1`;
    `?screen=0` turns it off.

- **The design** (`BLOCKS`, `SEQUENCES` in `src/config.js`); every block has new fractals:
  1. **Practice 1** (2×2): type I, on A or B at random.
  2. **Practice 2** (2×2): XOR.
  3. **Test** (4×4): three blocks of 208 rounds (13 passes of the 16 pairs), always in the
     order type VI → type X → type II:

     | Version | Block 1 | Block 2 | Block 3 |
     |---|---|---|---|
     | A (half the participants) | VI, A+AB | X, A+AB | II, A+B+AB |
     | B (the other half) | VI, B+AB | X, B+AB | II, A+B+AB |

     - Rules come from `kernel_model/kernel_modes.py` (`Design('4x4')`).
     - Type II weighs A and B equally, so it has no A/B version and block 3 is the same
       for both groups.
     - The version is random per participant, or fixed with `?version=A` / `?version=B`.
       For exactly half each, run two Prolific studies, one per version, and exclude each
       other's participants.

  Each practice block ends once 8 of the last 10 answers are correct
  (`PRACTICE_CRITERION`), or after 80 rounds at the latest (flagged in the data).

## Fractals

The symbols are made so that no fractal stands out and every block is equally hard to tell
apart. They come from [`../fractal_stimuli`](../fractal_stimuli), which documents how they
are made:

- **Matched and bright:** every fractal has the same lightness profile, CIE L* 70 / 56 / 78
  (outer / middle / inner layer), so each is clearly brighter than the grey background
  (L* 53.6). They also share chroma (60, lowered per hue to what a screen can show) and
  area.
- **Groups:** the set has 18 groups of 4 (test) and 12 pairs (practice). Within each
  group, every pairwise DreamSim distance (a perceptual similarity network) is within
  0.0075 of 0.257, so every group is equally hard.
- **No near-duplicates:** no two of the 96 fractals are closer than 0.161.

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
| <http://localhost:8000/> | **the real study**: full-length blocks (3 test blocks × 208 rounds, ~55 min). Use this for pilots whose data you want to analyse. |
| <http://localhost:8000/?debug=1&skip=intro,calibration> | a quick check that everything works (~3 min): shortened blocks, with a yellow "DEBUG RUN" banner at every block start |

Stop the server with Ctrl+C. After changing code, hard-reload the page (Cmd+Shift+R);
browsers cache the code.

Locally, add `?save=local`, and the data download at the end as
`catlearn_online_<date-time>_<tag>.csv` to your Downloads folder. Without it, a local run
uploads to the DataPipe Drive folder like a real session (see "Saving the data").

The page loads jsPsych from unpkg.com, so the pilot machine needs internet. Opening
`index.html` by double-clicking does not work; browsers block the code modules over
`file://`.

### URL options

| Option | Effect |
|---|---|
| `?debug=1` | **shortened blocks, for testing only**: test blocks 1 pass (16 rounds instead of 208), practice blocks at most 12 rounds. Every block start shows a DEBUG RUN banner, and the data have `debug = true` |
| `?skip=intro` | skip consent, instructions and the quiz |
| `?skip=calibration` | skip the card / blind-spot measurement (assumes 60 cm and a 96-dpi screen); combine as `?skip=intro,calibration` |
| `?seed=123` | fixed design (version, practice rule, fractals, round order) |
| `?version=A` | fixed test sequence version: `A` (A+AB in blocks 1–2) or `B` |
| `?save=local` | download the data instead of uploading them (testing without adding a session to the dataset) |
| `?simulate=1` | jsPsych plays the whole experiment by itself and downloads the data (never uploads) |
| `?simulate=visual` | the same, but at real speed on screen |

Options combine, e.g. `?debug=1&skip=intro,calibration&seed=5`. With no `PROLIFIC_PID` in
the URL the run counts as a pilot (`participant = pilot`) and never redirects to Prolific.

## Sizes in degrees

Every position and size in `LAYOUT` (`src/config.js`) is in degrees of visual angle. The
symbols are drawn at the EEG task's scale: 4° is the square that encloses the MATLAB task's
fractals. The new fractals have the same median extent and area; the spikiest reach 4.3°.
They are centred 6° left and right of the centre. The bull's eye (between rounds) is 0.6°
(0.2° centre, 0.15° cross).

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
- **Window too small.** If the whole layout (19° × 7°: both symbols and their feedback
  circles) does not fit the window, everything shrinks by the same factor (`layout_scale` < 1).
- **Why not webcam eye tracking?** It can't measure viewing distance reliably; the blind
  spot is the standard online method.

## Changing the design

Everything is in [`src/config.js`](src/config.js):

| Setting | What it holds |
|---|---|
| `RULES` | category 1 (F) / 0 (J) of every pair, for each 2×2 and 4×4 rule used. Generated from the task definitions in `kernel_model/kernel_modes.py`, the same as the task-type figures; pair = size · a + b. |
| `SEQUENCES` | the test rules in order, for version A and version B |
| `BLOCKS` | the blocks in order: phase, rule (a key, a list to pick from at random, or `seq:1` … `seq:3` for the participant's sequence), passes, and whether a practice criterion ends the block |
| `PRACTICE_CRITERION` | the window and number correct that end a practice block |
| `BONUS` | which phases count, and the per-block bonus tiers |
| `SCREENING`, `COMPREHENSION` | which checks screen participants out, the screen-out payment, quiz attempts |
| `TIMING`, `KEYS`, `LAYOUT`, `FEEDBACK` | durations (ms), the F/J keys, positions and sizes in degrees, feedback colours |
| `CALIBRATION` | blind-spot repetitions, plausible ranges, the values assumed when a measurement fails, fit margin |
| `BROWSER` | minimum window size; phones and tablets are excluded |
| `PROLIFIC` | completion codes |
| `DATA` | where data go |

The instructions compute their examples (bonus amounts, session length) from these settings, so
they stay correct when the design changes.

## Files

| File | What it does |
|---|---|
| `index.html` | loads jsPsych 8.3.0 and plugins (pinned versions), then `src/main.js` |
| `src/config.js` | all settings (above) |
| `src/design.js` | a participant's whole design from one seed: version, rules, fractals, round order. Pure functions; the same seed always gives the same design |
| `src/calibration.js` | card + blind-spot calibration, pixels per degree, fit to the window |
| `src/bonus.js` | accuracy → bonus in dollars: the one place the bonus rule is computed |
| `src/display.js` | HTML for every screen (symbols, bull's eye, feedback circles and word) in degrees |
| `src/task.js` | blocks: intro → rounds (iti → response → feedback) → block summary; practice criterion |
| `src/instructions.js` | consent, instructions, comprehension quiz (2 attempts, then screened out) |
| `src/data.js` | Prolific IDs from the URL, tab-switch / full-screen tracking, file name, local download (online saving is the DataPipe extension, set up in `src/main.js`) |
| `src/main.js` | puts it together: browser check → consent → full screen → calibration → instructions + quiz → blocks → final screen → save → Prolific |
| `tools/bonus_payments.py` | Prolific bonus list from the data files (see below) |
| `stimuli/fractal_groups/` | the 96 fractals the task uses, plus `groups.js` / `groups.json` (which fractals form each group) |

## Data

One CSV per participant, one row per screen. Analyse the rows with `part == "response"`
(one per round).

| Column | Meaning |
|---|---|
| `participant`, `prolific_pid`, `study_id`, `session_id` | Prolific IDs (`pilot` when run locally) |
| `seed`, `task_version` | regenerate the exact design with `buildDesign(seed, {version})` |
| `version`, `sequence` | the participant's test version (`A` or `B`) and the test types in order (`VI-X-II`) |
| `key_cat1`, `key_cat0` | the keys for category 1 and 0 (`f`, `j` for everyone) |
| `block`, `phase`, `block_in_phase`, `rule`, `rule_type`, `size`, `counts_for_bonus` | block information |
| `trial_in_block`, `rep` | round number in the block, and which pass through the pairs |
| `compound`, `level_a`, `level_b` | which pair: `compound = size · a + b` |
| `fractal_a`, `fractal_b` | fractal file numbers shown left (A) and right (B) |
| `category`, `correct_key` | the right answer: 1 / `f` or 0 / `j` |
| `response`, `choice`, `rt` | key pressed (lowercase), the answer (1 = F, category 1), and RT in ms from symbol onset (`null` if late) |
| `correct`, `timeout` | outcome |
| `recent_correct` | correct answers among the last 10 rounds (the practice criterion) |
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
| `block_summary` | `block_rounds`, `block_pcorrect`, `block_n_late`, test accuracy so far; test blocks `block_bonus_usd`, `bonus_so_far_usd`; practice `practice_criterion_met` |
| `screen_check`, `comprehension_check` | the screening checks; a screened-out participant has `screened_out` (`quiz`, `practice_type_I`, `practice_timeouts`) on every row |
| `final` | `test_pcorrect`, `test_correct`, `test_rounds`, `block_bonuses_usd`, **`bonus_usd`**, the complete tab-switch / full-screen log (`interaction_log`), `finished_at` |
| `iti`, `feedback` | timing checks |

Possible exclusion criteria to decide before data collection:

- `screened_out` set (already handled on Prolific)
- `practice_criterion_met == false` for the XOR practice
- many timeouts
- high `blur_count`
- `low_refresh`
- `calibration_source` other than `measured`, or a small `layout_scale`

## Saving the data

The study is live at `https://www.mindemory.io/catlearn_eeg/catlearn_4x4_prolific/` (GitHub
Pages from this repository's `main` branch; a push updates it within about a minute).

Data go through DataPipe (experiment `catlearn_4x4`, ID `PbuWRFfUHDfo`) to the Google Drive
folder `~/My Drive/DataPipe/catlearn_4x4/`, using DataPipe's jsPsych extension (loaded in
`index.html`; `DATA` in `src/config.js`):
- **Completed sessions:** trials are staged on DataPipe as the session runs, and the whole
  CSV is uploaded at the end.
- **Participants who quit partway:** they leave a `<file>-<id>.partial.json`, which doesn't
  count as a session.
- **File names carry no participant label:** `catlearn_online_<start time UTC>_<tag>.csv`.
  The Prolific ID and the file name (`session_file`) are inside the data.
- **If the final upload fails,** a copy downloads on the participant's computer, and the end
  screen asks them to send it through Prolific or by email, so they can be paid.
- **Test runs:** simulated runs and `?save=local` runs never upload.
- **Dashboard:** "Accept new data" must be on. Set "Stop after a set number of sessions" a
  little above your target, and switch off base64 uploads (not used).
- **Analysis:** `behav_4x4_prolific` copies new files from the Drive folder before every run
  (`--no-sync` skips this).

## Paying the bonus

Prolific pays bonuses separately from the base pay, after you approve submissions.

1. Make sure all finished sessions' CSV files are in one folder, e.g. the Drive folder
   itself, or `~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/` after an analysis run.
2. Run the payment script:

   ```bash
   python3 tools/bonus_payments.py ~/My\ Drive/DataPipe/catlearn_4x4
   ```

   It recomputes every bonus from the test rounds' accuracy, checks it against the `bonus_usd` the
   experiment showed and saved, and writes two files:
   - `bonus_payments.txt` (`PROLIFIC_PID,amount` per line)
   - `bonus_report.csv` (every file with its checks)

   Pilots (no Prolific ID) and files from the slot-machine pilot builds (task_version 3.x) are
   skipped. Screened-out sessions get no bonus (Prolific pays their fixed screen-out reward).
   Only finished test blocks earn a bonus. Duplicate IDs are paid once and flagged.
   Unfinished sessions are left out unless you pass `--include-incomplete`.
3. On Prolific, open the study → Submissions → **Bulk bonus payments**, and paste the
   contents of `bonus_payments.txt`.

If you change `BONUS.tiers` in `src/config.js`, change `TIERS` in the script to match.

## Going live on Prolific: checklist

1. **Hosting and data.** Done: GitHub Pages and DataPipe (see "Saving the data"). Run one
   full session from the public URL and check that the CSV arrives in the Drive folder.

2. **Prolific study URL.** Your hosted address followed by
   `?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`.
   Prolific fills in the IDs.
3. **Completion codes.** Copy each study's code from its "Completion paths" into
   `PROLIFIC.completionCodes` (`A` for the study whose URL has `&version=A`, `B` for the
   other); participants are redirected with their study's code automatically after saving.
   The URL's `version` must match the study, or the code won't be accepted.
4. **Custom screening.** In each study, under Data collection → Custom screening, choose
   "Yes":
   - screen-out reward $1.00, matching `SCREENING.payUsd`;
   - screen-out slots about half the places, e.g. 5 for 10;
   - copy its screen-out code into `PROLIFIC.screenOutCodes`, A and B.
5. **Device.** Restrict to desktop (laptop / computer), and set the estimated time: about
   55 minutes with the default design (3 × 208 test rounds at ~4.2 s, practice, ~8 min
   of setup).
6. **Study description.** Say that participants need a **bank or ID card** for the screen
   setup. Mention the bonus ("up to $6, per block") and that the study may end early after
   the instructions or the first practice, paid $1.00. Budget up to $6 of bonus per
   participant.
7. **Consent.** Check that the text in `src/instructions.js` matches the IRB-approved
   version, including the performance bonus.
8. **Two studies for the two versions.** For exactly half A and half B, publish the study
   twice, with `&version=A` and `&version=B` added to the URL, and exclude each other's
   participants.
9. **Pilot first.** Check it yourself with `?debug=1`, then run the real study once in full, then on Prolific with
   a handful of participants, checking each CSV and a test bonus payment before launching
   the rest.

## History

- **Version 1.0 (current, the first study): F/J categorisation.**
  - The slot machine, coins and coin bar are gone. Participants press F for one category
    and J for the other, with the same keys for everyone.
  - Rounds run automatically: 1 s fixation, then the pair with free viewing (no
    fixation while the symbols are on), then feedback.
  - The test is a fixed sequence, VI → X → II, in an A or a B version.
  - The bonus is paid per test block in tiers ($0.50 / $1.00 / $2.00, up to $6).
  - Screening: the quiz has 2 attempts. A failed quiz, a failed type-I practice or too
    many practice timeouts screen out with $1.00.
- **Pilot builds 3.0–3.2 (task_version 3.x): slot machine.** The participant pulled a lever (SPACE) and predicted
  win or lose (YES key randomised per participant). Coins: +10 correct, −5 wrong or late,
  500 coins = $1. Each participant got one test type in 3 blocks, with an A → B switch.
  - 3.1: equidistant fractal groups.
  - 3.2: DataPipe saving.

## Changes from the PsychoPy version

- **Categorisation.** The Friend-or-Enemy flags became an F/J categorisation of symbol pairs
  with a performance bonus (a slot machine with coins in versions 3.x).
- **Two symbols.** Two symbols (left / right) instead of three in a triangle. A 2×2 practice
  ends on a performance criterion, followed by three 4×4 test blocks.
- **Sizes in degrees.** Instead of fractions of the screen height: 4° symbols at 6°, as in
  the EEG task, with a per-participant screen calibration.
- **Feedback.** Green or red circles around the symbols, and a short word in the centre.
- **Fixation.** The MATLAB task's bull's eye replaces the cross, shown between rounds only
  (shown throughout in the slot-machine pilot builds). The PsychoPy file had its cross disabled,
  although the instructions asked participants to look at it.
- **Keys.** `F` / `J` (either case); PsychoPy used `E` / `F`.
- **Text instead of slides.** Instructions are HTML text rather than Keynote slides, and a
  comprehension quiz was added. The PsychoPy file also listed two instruction slides that
  don't exist (`instruction.008/.009`).
- **Eye tracking.** The calibration and fixation-check routines were dropped (lab only).
- **Logging.** Added Prolific IDs, the design seed, a browser check, full screen, tab-switch
  and full-screen-exit counts, and save/redirect handling.
