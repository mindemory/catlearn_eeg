# catlearn_task

MATLAB / Psychtoolbox-3 task for the category-learning EEG study: a 2×2 category
task with frequency-tagged (SSVEP) fractals. Code organization follows
`srh_ori/oriTask`: a top-level task script plus `initX` / `drawTextures` /
`showprompts` / `va2pixel` helpers, with per-block `timeReport` and `respReport`
structs saved to a `.mat` file. The flicker follows `ExampleSSVEP.m`.

## Design

Two fractals flank fixation on the horizontal meridian at 6 dva, each 4 dva wide.
Location **A** shows one of two fractals (a1 / a2) and location **B** one of two others
(b1 / b2), giving 4 compounds, `s = 2*levelA + levelB + 1`. The category (F or J)
depends on the block's rule, which subjects learn from feedback:

| Rule | Category |
|---|---|
| `A` | which A fractal (a1 vs a2) |
| `B` | which B fractal (b1 vs b2) |
| `AB` | A xor B |

Each subject does 3 blocks of 40 trials with one rule sequence: **A → A → AB**,
**A → B → AB** or **AB → AB → A**. Every block brings 4 new fractals (12 per
subject, none repeated), so each block's fractal-to-category mapping is learned
from scratch. The rule changes without warning.

Each fractal flickers with sinusoidal contrast at its own frequency, **12 or 15 Hz**,
so the SSVEP at each frequency indexes attention to that location. At 60 Hz this is
exactly 5 and 4 frames per cycle, and the 2 s stimulus holds whole cycles of both
(24 and 30). Contrast is set per frame, so the flicker stays frame-locked.

## Tasksets (make these offline, before the session)

```matlab
cd catlearn_task
makeTaskset(1)          % subject 01
makeTaskset(1:24)       % one full counterbalancing cycle
makeTaskset(3, true)    % regenerate subject 03, overwriting its file
```

`makeTaskset` writes `~/Documents/data/catlearn_eeg/catlearn_task/tasksets/taskset_subNN.mat`
and prints a one-line summary per subject. `catlearn_task` refuses to run without it.

- **Random per subject:** 12 fractals out of 72, 4 new per block
  (`taskset.fractals` is 3 × 4, rows = blocks, columns = a1 a2 b1 b2); per-block trial order (each
  compound 10 times, at most 3 identical compounds and 4 identical correct keys in a
  row); ITI jitter.
- **Counterbalanced by subject ID** (24 cells, repeating every 24 IDs; the sequence
  varies fastest): rule sequence (3) × side of A (left / right) × frequency
  assignment (12 Hz left / 15 Hz right, or the reverse) × keys (category 0 = F, or
  category 0 = J).
- **Reproducible:** everything random comes from `rng(1000 + ID, 'twister')`, so the
  same ID gives the same taskset on any machine.

## Running it

```matlab
cd catlearn_task
catlearn_task(1)        % subject 01, blocks 1-3, full-screen
catlearn_task(1, 2)     % resume at block 2 (e.g. after a crash)
catlearn_task(1, 1, 1)  % debug mode: transparent window, SkipSyncTests, keyboard not captured
testStimuli             % display check: geometry, tagging frames, one flicker with missed-frame count
```

Between blocks, SPACE continues and ESCAPE ends the session. ESCAPE during a
stimulus or response window aborts; everything collected up to that point is saved.

Psychtoolbox is auto-added from `~/Documents/MATLAB/Psychtoolbox/Psychtoolbox` (the
toolbox folder inside a git clone of Psychtoolbox-3), `/Applications/Psychtoolbox` or
`/usr/share/psychtoolbox-3` if it is not already on the path.

On macOS and Windows, PTB 3.0.20+ requires license management (a paid key, or a
14-day trial per machine). Linux PTB needs neither. Flip timing on a Mac in debug
mode (transparent window, sync tests skipped) is not trustworthy: check flicker
timing on the rig with `testStimuli`.

## Trial structure

| Phase | Duration | On screen |
|---|---|---|
| Fixation | 1.0 s | fixation cross |
| Stimulus | 2.0 s | both fractals flickering (12 / 15 Hz), **cross stays up** |
| Response | ≤ 2.0 s from stimulus **offset** | fixation cross; F or J |
| Feedback | 0.5 s | `Correct` (green) / `Incorrect` (red) / `Too slow` (amber) |
| ITI | 1.0–1.5 s jittered | dimmed fixation |

About 6 s per trial, about 4 min per block.

- **Responses count only after the flicker.** F/J pressed during the stimulus are
  ignored and logged as `earlyPress`, so every trial has a full, response-free
  2 s SSVEP epoch.
- **RT is measured from stimulus offset** (`RT`); `RTfromOnset` is also saved.
- Each flicker starts at phase 0 (both fractals at zero contrast) on the flip that
  sends the stimulus-onset trigger.

## Stimuli

- Every fractal PNG (500 × 500 RGBA) is cropped to one shared square: the smallest
  square enclosing the visible part of all 72 fractals. That square is drawn 4 dva
  wide, so fractals keep their relative sizes.
- Contrast is the texture's global alpha. Over the grey background, alpha blending
  scales each fractal's deviation from grey. On a non-linearized monitor, luminance
  contrast is not linear in this value.

## Files

| File | Purpose |
|---|---|
| `catlearn_task.m` | main task: session and block loop, flicker, response collection, saving |
| `makeTaskset.m` | **offline** per-subject taskset: fractals, counterbalancing, trial orders |
| `loadParameters.m` | all timing, stimulus, design, color and EEG settings |
| `makeTaskMap.m` | trial table for one block, read from the taskset |
| `initScreen.m` | opens the PTB window, computes px↔cm↔dva factors |
| `initPeripherals.m` | finds the keyboard, creates a KbQueue restricted to F/J/space/escape |
| `initStimuli.m` | fractal textures, rects and per-frame flicker contrast, built **once** per session; checks the refresh rate |
| `initFiles.m` | per-block output directory and file names |
| `drawTextures.m` | draws fixation / fractals into the backbuffer (never flips) |
| `showprompts.m` | all on-screen text |
| `sendTrigger.m` | single choke point for EEG triggers (currently a no-op stub) |
| `va2pixel.m`, `pixel2va.m` | visual angle ↔ pixel conversion |
| `testStimuli.m` | standalone display, geometry and flicker-timing check |
| `ExampleSSVEP.m` | reference SSVEP script the flicker is based on (not used by the task) |
| `fractals/` | the 72 fractal PNGs (`generate_fractals.py`) |

## Output

Written outside the repo, to `~/Documents/data/catlearn_eeg/catlearn_task/sub<ID>/block<NN>/`
(`parameters.dataDir` in `loadParameters.m`; built from `$HOME`, so the same path
works on the Mac and the Linux rig):

- `matFile_subj<ID>_block<NN>.mat` → `matFile.{parameters, screen, stim, taskset, tMap, timeReport, respReport, aborted, nTrialsCompleted}`
  - `respReport(trial)`: `key`, `correctKey`, `RT` (s from stimulus offset),
    `RTfromOnset`, `correct`, `timedOut`, `earlyPress`, `rule`, `stim`, `levelA`,
    `levelB`, `category`
  - `timeReport(trial)`: flip timestamps and achieved durations, including
    `flipTimes` (every stimulus frame) and `missedFrames`. Check these to confirm
    the flicker kept up.
  - `stim`: geometry, `freqLeftRight`, `freqActual`, `framesPerCycle`, `contrast`
    (per-frame contrast, left/right), `sideA`, `fractals` (all blocks; this block's
    are also in `tMap.fractals`)
- `EEGreportFile_subj<ID>_block<NN>.mat` → `EEGsummary.{timeReport, trigReport}`

## EEG triggers

`parameters.EEG = 0` for now. Every event already calls `sendTrigger`, which
timestamps the event and returns without touching hardware, so turning triggers on
requires no changes to `catlearn_task.m`:

1. Set `parameters.EEG = 1` and `parameters.triggerMethod` in `loadParameters.m`.
2. Implement the matching branch in `sendTrigger.m` (`parallel`, `serial` or `python`).

`catlearn_task` errors at startup if `EEG = 1` while `triggerMethod` is `'none'`, so
a misconfiguration fails before the first trial rather than mid-block.

| Code | Event |
|---|---|
| 0 | block start |
| 1 | fixation onset |
| 21–24 | stimulus (flicker) onset, compound 1–4 |
| 3 | stimulus offset = response window onset |
| 41 / 42 | response `F` / `J` |
| 51 / 52 / 53 | feedback correct / incorrect / timeout |
| 6 | ITI onset |
| 7 | block end |

Rule, side and frequencies are fixed per block and subject, so the trigger plus the
saved taskset identify every trial. The `python` (shell out per trigger) approach
carried over from `srh_ori` costs tens of milliseconds with poor jitter. Prefer
parallel or serial for EEG, especially with the tagging onsets.

## Before running subjects

- `parameters.viewingDistance` in `loadParameters.m` defaults to 55 cm. Set it to the
  actual rig distance, since every dva→pixel conversion depends on it.
- Check the rig's refresh rate. 12 / 15 Hz needs 60 or 120 Hz; `initStimuli` errors
  (outside debug mode) if a frequency is not a whole number of frames.
- Run `testStimuli` and check the printed pixels/degree, fractal size, eccentricity,
  frames per cycle and **missed frames**.
- `initScreen` reads physical screen size from EDID via `Screen('DisplaySize')`, which
  some displays report incorrectly. Measure the panel and hard-code
  `screen.screenWidth` / `screen.screenHeight` if the printed cm values look wrong.
- `initPeripherals` takes the first keyboard `GetKeyboardIndices` reports. On the rig,
  set `parameters.kbName` to pin the response device.
- Make the tasksets (`makeTaskset(1:N)`) and copy the `tasksets/` folder to the rig
  (or regenerate there: same IDs give identical tasksets).

## Online version

The browser (jsPsych) version for Prolific, a 4×4 slot-machine task with a coin bonus,
lives next to this folder in [`../catlearn_4x4_prolific/`](../catlearn_4x4_prolific/). Its
README covers local piloting, the data columns, paying bonuses and the Prolific checklist.
