// jsPsych plugin for one trial: fixation, the two patches, the response window.
//
// Frame-accurate timing. The whole screen (fixation and both patch canvases, already
// drawn) is built at the start of the fixation period with the patches hidden; onset and
// offset only switch their visibility. Building a screen (layout, canvas upload) can make
// the browser miss a frame, which would delay the onset unpredictably; switching visibility
// is cheap. Each switch is made in a frame callback; the change appears at the next
// frame, so onset and offset are timestamped at the callback after the switch. Exposure =
// offset - onset (performance.now ms, from frame timestamps).
//
// Responses count from the onset frame until response_ms after the offset.

import { drawField } from './noise.js';

const { ParameterType } = jsPsychModule;

const info = {
  name: 'noise-pair',
  version: '1.0.0',
  parameters: {
    html: { type: ParameterType.HTML_STRING, default: undefined },   // screen with hidden canvases #patch-left/right
    fields: { type: ParameterType.COMPLEX, default: undefined },     // {left, right} noise fields (pass a function: made at trial start)
    choices: { type: ParameterType.KEYS, default: undefined },
    fixation_ms: { type: ParameterType.INT, default: 500 },
    stimulus_ms: { type: ParameterType.INT, default: 100 },
    response_ms: { type: ParameterType.INT, default: 2000 },
  },
  data: {
    response: { type: ParameterType.KEY },
    rt: { type: ParameterType.INT },                     // ms from onset; null if late
    fixation_measured_ms: { type: ParameterType.FLOAT },
    stim_ms: { type: ParameterType.FLOAT },
    stim_frames: { type: ParameterType.INT },           // frames shown, counted by frame callbacks
    stim_max_gap: { type: ParameterType.FLOAT },        // longest gap between callbacks while shown (> 1.5 periods: a skipped frame)
    stim_hidden_by: { type: ParameterType.STRING },
    answered_during_stimulus: { type: ParameterType.BOOL },
  },
};

class NoisePairPlugin {
  static info = info;

  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display, trial) {
    display.innerHTML = trial.html;
    const canvases = ['left', 'right'].map((s) => display.querySelector(`#patch-${s}`));
    const { fields } = trial;
    drawField(canvases[0], fields.left);
    drawField(canvases[1], fields.right);
    canvases.forEach((c) => { c.style.visibility = 'hidden'; });

    // phase: fix -> showing (switched on, not yet presented) -> on -> hiding -> off
    const s = { phase: 'fix', first: null, onset: null, offset: null, frames: 0, last: null, maxGap: 0,
                hiddenBy: null, response: null, rt: null, done: false, keyStart: null };
    const setVisible = (v) => canvases.forEach((c) => { c.style.visibility = v ? 'visible' : 'hidden'; });

    const end = () => {
      if (s.done) return;
      s.done = true;
      this.jsPsych.pluginAPI.cancelAllKeyboardResponses();
      this.jsPsych.pluginAPI.clearAllTimeouts();
      display.innerHTML = '';
      this.jsPsych.finishTrial({
        response: s.response,
        rt: s.rt,
        fixation_measured_ms: s.onset !== null && s.first !== null ? s.onset - s.first : null,
        stim_ms: s.onset !== null && s.offset !== null ? s.offset - s.onset : null,
        stim_frames: s.onset === null ? null : s.frames + 1,
        stim_max_gap: s.maxGap,
        stim_hidden_by: s.hiddenBy,
        answered_during_stimulus: s.response !== null && s.offset === null,
      });
    };

    const startResponses = (t0) => {
      s.keyStart = t0;
      this.jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: (info) => {
          s.response = info.key;
          s.rt = Math.round(performance.now() - s.onset);   // from the onset frame
          end();
        },
        valid_responses: trial.choices,
        rt_method: 'performance',
        persist: false,
        allow_held_key: false,
      });
    };

    const hide = (by) => {
      setVisible(false);
      s.phase = 'hiding';
      s.hiddenBy = by;
    };

    const step = (now) => {
      if (s.done) return;
      if (s.first === null) s.first = now;
      if (s.phase === 'fix' && now - s.first >= trial.fixation_ms - 5) {
        setVisible(true);                          // presented at the next frame
        s.phase = 'showing';
      } else if (s.phase === 'showing') {
        s.onset = now;                             // this frame shows the patches
        s.last = now;
        s.phase = 'on';
        startResponses(now);
        this.jsPsych.pluginAPI.setTimeout(end, trial.stimulus_ms + trial.response_ms);
        // backup in case frame callbacks stall (e.g. a hidden page): off at most 50 ms late
        this.jsPsych.pluginAPI.setTimeout(() => { if (s.phase === 'on') { hide('timer'); s.offset = performance.now(); s.phase = 'off'; } },
                                          trial.stimulus_ms + 50);
      } else if (s.phase === 'on') {
        s.frames += 1;
        s.maxGap = Math.max(s.maxGap, now - s.last);
        s.last = now;
        // switch off when the next frame would reach the target exposure
        if (now - s.onset >= trial.stimulus_ms - 5 - (now - s.onset) / Math.max(1, s.frames)) hide('frame');
      } else if (s.phase === 'hiding') {
        s.offset = now;                            // this frame no longer shows them
        s.maxGap = Math.max(s.maxGap, now - s.last);
        s.phase = 'off';
        return;
      } else if (s.phase === 'off') {
        return;
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    // failsafe: never hang if frame callbacks stop before the onset (e.g. a hidden page)
    this.jsPsych.pluginAPI.setTimeout(end, trial.fixation_ms + trial.stimulus_ms + trial.response_ms + 1000);
  }

  // Simulation: a random key after a random RT (data-only: no screen; visual: the real trial)
  simulate(trial, simulationMode, simulationOptions, loadCallback) {
    const rt = Math.round(this.jsPsych.randomization.sampleExGaussian(600, 80, 1 / 150, true));
    const key = this.jsPsych.pluginAPI.getValidKey(trial.choices);
    if (simulationMode === 'data-only') {
      loadCallback();
      this.jsPsych.finishTrial({
        response: key, rt, fixation_measured_ms: null, stim_ms: null, stim_frames: null, stim_max_gap: null,
        stim_hidden_by: null, answered_during_stimulus: false,
        ...simulationOptions.data,
      });
      return;
    }
    this.trial(this.jsPsych.getDisplayElement(), trial);
    loadCallback();
    this.jsPsych.pluginAPI.setTimeout(() => this.jsPsych.pluginAPI.pressKey(key), trial.fixation_ms + 30 + rt);
  }
}

export default NoisePairPlugin;
