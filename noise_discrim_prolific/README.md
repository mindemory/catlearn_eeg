# noise_discrim_prolific: same / different noise patches (jsPsych)

How well can people tell two 1/f^alpha noise patches apart, as a function of alpha? This
online study measures sensitivity (d') and bias (c) at 10 alphas from 0 to 4. It uses the
noise patches of the 2×2×2 learning task ([`../catlearn_2x2x2_prolific`](../catlearn_2x2x2_prolific)),
seen at the same eccentricity and size. It runs on its own sample, so it can't affect
anyone's learning.

- **A trial (~1.6 s):**

  | Step | What happens | Duration |
  |---|---|---|
  | fixation | the bull's eye alone | 0.5 s |
  | patches | two patches, 6° left and right of fixation, 4° across | 100 ms (6 frames at 60 Hz) |
  | response | fixation alone; `F` / `J` = same / different (either case) | up to 2 s after the patches |
  | feedback | the fixation's centre turns green (correct) or red (wrong or late) | 0.3 s |

  - **Same or different:** both patches have the same alpha. On *same* trials they are the
    very same image; on *different* trials they are two different images.
  - **Keys:** which key means SAME is randomised per participant.
  - **Early answers:** answers during the patches count, and end the trial.
  - **Why 100 ms:** too short to move the eyes to a patch, so viewing stays peripheral even
    without eye tracking.
- **Design** (`ALPHAS`, `BLOCKS` in `src/config.js`). Method of constant stimuli:
  - **Levels:** 10 alphas, 0 to 4 in steps of 4/9 (0, 0.44, 0.89, 1.33, 1.78, 2.22, 2.67,
    3.11, 3.56, 4.0).
  - **Blocks:** 5 blocks of 80 trials (`BLOCKS`). Each block has 4 same and 4 different
    trials of every alpha, in random order. In all, 400 trials: 20 same + 20 different per
    alpha. Near d' = 1, one participant's d' at one alpha is then about ±0.4 (it was ±0.6
    with 10 + 10); the group mean is much tighter.
  - **No practice:** the instructions show an example same and different pair at the real
    size.
- **Time:** about 11 min of trials plus breaks, about 17 min with calibration and
  instructions.
- **Pay:** fixed; there is no performance bonus.

## Stimuli

Patches are generated in the browser (`src/noise.js`), using the recipe of
`task_design/noise_patches.py`:
1. white noise;
2. FFT, amplitude k^−alpha;
3. inverse FFT;
4. RMS contrast 0.3 of the [−1, 1] field;
5. clip to [−1, 1];
6. circular aperture with a raised-cosine edge, shown as grey 0.5 + 0.5 · field ·
   aperture.

Each patch is a 128 × 128 grid stretched to 4°.

- **One addition: a spectral cutoff at 8 cycles/deg** (`NOISE.cutoffCpd`). Without it, the
  finest detail in low-alpha patches would depend on each participant's screen resolution.
  8 c/deg is about the finest detail resolvable at 6° eccentricity. The cutoff matters only
  at low alpha; of the variance a full-band patch would have, it removes:

  | alpha | 0 | 0.89 | 1.5 | 2 and above |
  |---|---|---|---|---|
  | removed | 80% | 25% | 1% | < 0.1% |

  So from alpha ~1.5 on, the patches match the learning task's.
- **Large alphas:** at alpha 3–4 a patch is mostly one smooth gradient or blob. Its mean
  brightness then differs from the background, which is a natural property of these
  patches.
- **Every stimulus can be regenerated.** Each trial saves its alpha and two seeds
  (`seed_left`, `seed_right`, equal on same trials). `tools/noise_field.py` reproduces the
  browser's patches exactly: the same random-number generator and the same order of
  draws, equal to 10 decimal places (checked).

  ```bash
  python3 tools/noise_field.py 2.6667 123456789 --out patch.png
  ```

- **Timing** (`src/plugin-noise-pair.js`, a small jsPsych plugin that runs fixation,
  patches and response as one trial):
  - **Screen built in advance:** the whole screen, both patches drawn but hidden, is built
    at the start of the fixation period.
  - **Cheap onset and offset:** onset and offset only switch the patches' visibility.
    Building a screen (layout, uploading the canvases) can make the browser miss a frame,
    which delayed the onset unpredictably in version 1.0. Switching visibility is cheap.
  - **Timestamped when shown:** each switch is made in a frame callback and shows at the
    next frame, so onset and offset are timestamped there. `stim_ms` is the exposure
    actually shown.
  - **Checked with a simulated display:**
    - 100.0 ms at 60 and 120 Hz;
    - the nearest whole number of frames elsewhere (97 ms at 144 Hz, 107 ms at 75 Hz);
    - unchanged by a skipped frame mid-stimulus.
  - **Skipped frames:** `stim_frames` counts frame callbacks, so it reads one low when the
    browser skips one; `stim_max_gap` (the longest gap between callbacks) flags that. The
    exposure stays right.
  - If the browser stops drawing frames (e.g. the tab is hidden), a backup timer removes
    them at most 50 ms late, and the trial is marked `timer`.

## Run it locally (pilot)

```bash
cd ~/Documents/codes/catlearn_eeg/noise_discrim_prolific
```

```bash
python3 -m http.server 8000
```

| Address | What you get |
|---|---|
| <http://localhost:8000/> | **the real study** (~17 min) |
| <http://localhost:8000/?debug=1&skip=intro,calibration> | quick check (~1 min): 2 blocks of 20 trials, with a DEBUG RUN banner |

The data download as `noise_discrim_pilot_<date-time>.csv` at the end. The page loads
jsPsych from unpkg.com, so it needs internet. Opening `index.html` by double-clicking does
not work; browsers block the code modules over `file://`.

| Option | Effect |
|---|---|
| `?debug=1` | 2 blocks of 20 trials (1 same + 1 different per alpha) |
| `?skip=intro` | skip consent, instructions and the quiz |
| `?skip=calibration` | skip the card / blind-spot measurement (assumes 60 cm and a 96-dpi screen) |
| `?seed=123` | fixed design (keys, trial order, patch seeds) |
| `?id=alex` | label a run outside Prolific (saved as `participant`, and in the file name) |
| `?save=local` | download the data instead of uploading them (testing the hosted page) |
| `?simulate=1` | jsPsych plays the whole study by itself and saves the data |
| `?simulate=visual` | the same, on screen |

## Sizes and calibration

Positions and sizes (`LAYOUT`) are in degrees of visual angle. Patches are 4° across,
centred 6° left and right of fixation; the bull's eye is 0.6°.

Pixels per degree come from jsPsych's virtual chinrest: a card-sizing step, then the blind
spot. This is as in the learning tasks; see the
[4×4 README](../catlearn_4x4_prolific/README.md#sizes-in-degrees). If the 18° × 6° layout
doesn't fit the window, everything shrinks by the same factor (`layout_scale`).

## Files

| File | What it does |
|---|---|
| `index.html` | loads jsPsych 8.3.0 and plugins (pinned versions), then `src/main.js` |
| `src/config.js` | all settings: alphas, blocks, noise recipe, layout, timing, keys, calibration, Prolific |
| `src/design.js` | a participant's design from one seed: key mapping, trial order, patch seeds |
| `src/noise.js` | seeded noise patches (PRNG, FFT, 1/f^alpha filter, aperture, drawing) |
| `src/display.js` | trial screens (fixation, patch canvases) and the instruction examples, in degrees |
| `src/plugin-noise-pair.js` | jsPsych plugin for one trial: fixation, patches switched on and off in display frames, response window |
| `src/task.js` | blocks and trials (plugin trial → feedback); break screens |
| `src/instructions.js` | consent, instructions with example pairs, comprehension quiz |
| `src/calibration.js` | card + blind-spot calibration (as in the learning tasks) |
| `src/data.js` | Prolific IDs from the URL, tab-switch / full-screen tracking, saving |
| `src/main.js` | browser check → consent → full screen → calibration → instructions + quiz → blocks → final screen → save → Prolific |
| `tools/noise_field.py` | regenerates any trial's patches in Python |

Analysis: [`../behav_noise_discrim`](../behav_noise_discrim) (d' and c by alpha).

## Data

One CSV per participant, one row per screen. Analyse the rows with `part == "response"`
(one per trial).

| Column | Meaning |
|---|---|
| `participant`, `prolific_pid`, `study_id`, `session_id` | Prolific IDs (`pilot` when run locally) |
| `seed`, `task_version` | regenerate the design with `buildDesign(seed)` |
| `key_same`, `key_different` | this participant's keys |
| `block`, `trial_in_block` | position in the session |
| `alpha`, `alpha_level` | the spectral slope, and its level (1-10) |
| `pair` | `same` or `different` (the right answer) |
| `seed_left`, `seed_right` | patch seeds (`tools/noise_field.py`) |
| `response`, `answer`, `rt` | key pressed, `same` / `different`, RT in ms from the patches' onset (null if late) |
| `correct`, `timeout`, `said_different` | outcome |
| `stim_ms`, `stim_frames`, `stim_max_gap`, `stim_hidden_by`, `fixation_measured_ms` | measured exposure (see Timing) |
| `answered_during_stimulus` | answered before the patches went off |
| `blur_count`, `fullscreen_exit_count` | times the participant left the tab or full screen so far |
| `px_per_deg`, `layout_scale`, `window_width`, `window_height` | screen scale |

Other rows:

| `part` | What it records |
|---|---|
| `browser_check` | window size, browser, refresh rate (`low_refresh` if under 50 Hz) |
| `calibration_summary` | `calibration_source`, `px_per_mm`, `view_dist_mm`, `px_per_deg`, `layout_scale` |
| `comprehension` | quiz answers, `comprehension_passed` |
| `design` | the alphas and the noise settings |
| `block_start` | `prev_block_pcorrect` |
| `final` | `pcorrect`, the tab-switch / full-screen log, `finished_at` |

Exclusion flags, to fix before data collection (computed by the analysis):
- not above chance over alpha ≥ 2.22 (one-sided binomial test, p ≥ .05). No alpha is easy
  at 100 ms (the first full pilot reached about 71% correct above alpha 1.33), so this only
  checks that the participant did the task;
- the comprehension quiz never passed;
- more than 10% of trials with an unknown or wrong exposure.

## Putting it online

Prolific doesn't host studies or store data: it sends participants to a URL you give it.
Two things are needed: somewhere public to host this folder, and somewhere to send the
data. With `DATA.save = 'local'` the data download to the *participant's* computer, and
you never see them.

**1. Data: DataPipe to Google Drive** (set up). `DATA` in `src/config.js` sends each
session's CSV once, at the end, through DataPipe (experiment `noise_discrim_thresholding`,
ID `A6UGkxMHa4mJ`). DataPipe stores it in the Google Drive folder linked to that experiment.
- **Dashboard:** "Accept new data" must be on. Its check rejects files without a
  `trial_type` column (every jsPsych CSV has one).
- **If an upload fails** (network, or collection switched off), a copy downloads on the
  participant's computer and the end screen asks them to email it to `CONTACT.email`.
- **Test runs:** simulated runs (`?simulate=1`) never upload, and `?save=local` keeps any
  run offline, so you can test the hosted page without adding a file to the dataset.
- **Public ID:** the experiment ID is visible in the page source, as with any DataPipe
  study. To limit misuse, set "Stop after a set number of sessions" a little above your
  target, and switch off base64 uploads (not used here).
- **No partial data:** DataPipe's streaming extension would also save the trials of people
  who quit partway. It isn't used, because it sends a request after every trial of a
  frame-timed task.

**2. Hosting: any static web host.** The study is plain files: jsPsych loads from unpkg, and
the noise is generated in the browser, so nothing else is needed. Two easy options:
- **Netlify:** log in at [app.netlify.com](https://app.netlify.com), then *Add new site →
  Deploy manually*, and drag this folder in. You get a URL like
  `https://noise-discrim.netlify.app`; drag again to update.
- **GitHub Pages:** put this folder in its own repository (Pages is free only for public
  repositories), then *Settings → Pages → Deploy from branch*.

After deploying, run it once yourself from the public URL and check the file arrives in the Drive folder.

**3. Trying it with friends (no Prolific needed).** Send the hosted URL with a label, e.g.
`https://noise-discrim.netlify.app/?id=alex`. The label goes into `participant` and the file
name. Friends can't take a Prolific study unless they are Prolific participants who match
its screening, and Prolific would expect them to be paid. Without `PROLIFIC_PID` in the
URL, the study ends with a "Pilot finished" screen instead of redirecting to Prolific.

## Going live on Prolific

1. **Data and hosting.** As above, with a run from the public URL checked in the Drive folder first.
2. **Study URL.** Your hosted address followed by
   `?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`.
3. **Completion code.** Put it in `PROLIFIC.completionCode`.
4. **Device and time.** Restrict to desktop; estimated time 17 min.
5. **Description.** Mention the card needed for the screen setup.
6. **Consent.** Check the text in `src/instructions.js` against the IRB-approved version.
7. **Exclusions.** Exclude people who took part in the learning studies, and vice versa.
