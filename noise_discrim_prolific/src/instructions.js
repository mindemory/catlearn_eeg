// Consent, instructions and the comprehension check. The instructions come after the screen
// calibration, so their example patches are drawn at the real size and spacing.

import { COMPREHENSION, CONTACT, FEEDBACK, RECRUITMENT, SESSION_MINUTES, TIMING } from './config.js';
import { examplePair } from './display.js';
import { fieldDataUrl, noiseField } from './noise.js';

const key = (k) => `<span class="key">${k.toUpperCase()}</span>`;
const page = (html) => `<div class="page">${html}</div>`;

export function consentTrial(onDecline) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: page(`
      <h2>Research project information</h2>
      <p>This online experiment is conducted by researchers from the Department of Psychological and Brain
      Sciences at Dartmouth College, Hanover, NH, USA. The purpose of this study is to understand how well
      humans can tell visual patterns apart.</p>
      <p>During this study, you will complete a computer-based task that involves looking at patterns on the
      screen. On each trial, you will briefly see two patterns and will be asked to make a simple decision using
      your keyboard.</p>
      ${RECRUITMENT === 'pilot'
    ? `<p>This is a short pilot of the study, with a small number of volunteers. <b>Participation is unpaid:
      you will not receive any compensation.</b> It takes up to ${SESSION_MINUTES} minutes.</p>
      <p>You must be 18 or older to participate. Participation is entirely voluntary, and you may stop at any time,
      for any reason, by closing this window.</p>`
    : `<p>You must be 18 or older to participate. Participation is entirely voluntary, and you may stop at any time
      if you feel uncomfortable for any reason. If you withdraw early due to discomfort or technical issues,
      compensation may not be possible due to Prolific platform limitations.</p>`}
      <p>All information collected will be kept confidential and securely stored on Dartmouth's secure servers,
      accessible only to the principal investigator and authorized research staff. No identifying personal
      information will be collected.</p>
      <p class="small">Questions about this project may be directed to: ${CONTACT.name}, ${CONTACT.address},
      ${CONTACT.email}</p>`),
    choices: ['I agree, continue', 'I do not agree'],
    data: { part: 'consent' },
    simulation_options: { data: { response: 0 } },   // simulated participants consent
    on_finish: (data) => {
      data.consent = data.response === 0;
      if (!data.consent) onDecline();
    },
  };
}

/**
 * Instruction pages, then a short quiz; repeated (up to COMPREHENSION.maxAttempts) until
 * all answers are right.
 */
export function instructionsWithCheck(design) {
  const S = design.keys.same.toUpperCase();
  const D = design.keys.different.toUpperCase();
  const nBlocks = design.blocks.length;
  const nTrials = design.blocks[0].length;
  let examples = null;   // drawn on first use (needs a document)
  const ex = () => {
    if (!examples) {
      const { alpha, same, different } = design.example;
      const url = (s) => fieldDataUrl(noiseField(alpha, s));
      const sameUrl = url(same[0]);
      examples = examplePair([sameUrl, sameUrl], 'SAME: exactly the same picture twice')
        + examplePair([url(different[0]), url(different[1])], 'DIFFERENT: two different pictures of the same kind');
    }
    return examples;
  };

  const instructions = {
    type: jsPsychInstructions,
    pages: () => [
      page(`<h2>Same or different?</h2>
        <p class="center">On each trial two patterns flash briefly, one on each side of the centre. Sometimes they
        are <b>exactly the same picture</b>; sometimes they are <b>two different pictures</b> of the same kind of
        pattern. Your job is to tell which.</p>
        <div class="example-col">${ex()}</div>`),
      page(`<h2>Your answer</h2>
        <p class="center">Press ${key(S)} if the two pictures were the <b>SAME</b>,<br>
        ${key(D)} if they were <b>DIFFERENT</b>.</p>
        <p class="center">The pictures are shown for only a fraction of a second. Answer within
        ${TIMING.response / 1000} seconds; if you are not sure, make your best guess.
        ${FEEDBACK.show ? `The centre of the target then turns <b style="color: ${FEEDBACK.correct}">green</b> if
        you were right and <b style="color: ${FEEDBACK.wrong}">red</b> if you were wrong or too slow.` : ''}</p>
        <p class="center">Some trials are easy and some are very hard; that is expected.</p>`),
      page(`<h2>Keep your eyes on the centre</h2>
        <p class="center">Always keep your eyes on the small black-and-white target in the middle of the screen.
        The pictures appear to its left and right; they are too brief to look at directly.</p>`),
      page(`<h2>Procedure</h2>
        <ul>
        <li>First ${design.practice.length} practice trials (not scored), then ${nBlocks} blocks of ${nTrials}
          trials, with a short break after each block.</li>
        <li>Each trial starts by itself; keep your fingers on ${key(S)} and ${key(D)}.</li>
        <li>The whole study takes up to ${SESSION_MINUTES} minutes.</li></ul>
        <p class="center">Next, a few quick questions to check the instructions.</p>`),
    ],
    show_clickable_nav: true,
    allow_backward: true,
    key_forward: 'ArrowRight',
    key_backward: 'ArrowLeft',
    data: { part: 'instructions' },
  };

  const answers = {
    same_key: S,
    same_means: 'Exactly the same picture twice',
    look_where: 'At the target in the middle',
  };
  let attempts = 0;
  let passed = false;
  const quiz = {
    type: jsPsychSurveyMultiChoice,
    preamble: '<h3>Quick check</h3>',
    questions: [
      { name: 'same_key', prompt: 'Which key do you press if the two pictures were the <b>same</b>?',
        options: ['F', 'J'], required: true },
      { name: 'same_means', prompt: 'When are the pictures "the same"?',
        options: ['Exactly the same picture twice', 'Two pictures of the same kind of pattern'], required: true },
      { name: 'look_where', prompt: 'Where should you keep your eyes?',
        options: ['At the target in the middle', 'At the pictures on the sides'], required: true },
    ],
    data: { part: 'comprehension' },
    on_finish: (data) => {
      attempts += 1;
      passed = Object.entries(answers).every(([k, v]) => data.response[k] === v);
      data.comprehension_passed = passed;
      data.comprehension_attempt = attempts;
    },
    simulation_options: { data: { response: answers } },
  };
  const retry = {
    timeline: [{
      type: jsPsychHtmlButtonResponse,
      stimulus: page('<p class="center">Some answers were not right. Please read the instructions again.</p>'),
      choices: ['Show the instructions again'],
      data: { part: 'comprehension_retry' },
    }],
    conditional_function: () => !passed && attempts < COMPREHENSION.maxAttempts,
  };

  // After maxAttempts failures the participant continues anyway; comprehension_passed = false
  // in the data flags them for exclusion.
  return {
    timeline: [instructions, quiz, retry],
    loop_function: () => !passed && attempts < COMPREHENSION.maxAttempts,
  };
}

