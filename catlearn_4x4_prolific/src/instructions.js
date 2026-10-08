// Consent, instructions and the comprehension check. Text only here; the trial screens
// are in display.js and the task timeline in task.js.

import { BONUS, COMPREHENSION, CONTACT, FEEDBACK, practiceCriterion, QUESTIONNAIRE, SCREENING, TIMING } from './config.js';
import { exampleScreen } from './display.js';
import { formatDollars, maxBonus } from './bonus.js';

const key = (k) => `<span class="key">${k.toUpperCase()}</span>`;
const page = (html) => `<div class="page">${html}</div>`;

export function consentTrial(onDecline) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: page(`
      <h2>Research project information</h2>
      <p>This online experiment is conducted by researchers from the Department of Psychological and Brain
      Sciences at Dartmouth College, Hanover, NH, USA. The purpose of this study is to understand how humans
      use visual information in novel settings to perform optimal decision making.</p>
      <p>During this study, you will complete a computer-based task that involves looking at different shapes or
      patterns on the screen. On each trial, you will see a set of images and will be asked to make a simple
      decision using your keyboard. The task is designed to see how people notice, remember, and respond to
      different types of visual information.</p>
      <p>You must be 18 or older to participate. Participation is entirely voluntary, and you may stop at any time
      if you feel uncomfortable for any reason. If you withdraw early due to discomfort or technical issues,
      compensation may not be possible due to Prolific platform limitations.</p>
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
 * @param {object} design       from buildDesign
 * @param {number[]} example    two fractal numbers for the example screens
 * @param {{onFail?: function}} opts  onFail: called if every attempt fails (screening);
 *   without it the participant continues, flagged by comprehension_passed = false
 */
export function instructionsWithCheck(design, example, { onFail = null } = {}) {
  const practice = design.blocks.filter((b) => b.phase === 'practice');
  const test = design.blocks.filter((b) => b.phase === 'test');
  // e.g. 'Block 1 ends once 11 of your last 12 answers are correct, block 2 once 9 of your last 10.'
  const criteriaText = practice.filter((b) => b.criterion).map((b, i) => {
    const { window: w, minCorrect: m } = practiceCriterion(b.ruleType);
    return i === 0 ? `Block ${b.blockInPhase} ends once ${m} of your last ${w} answers are correct`
      : `block ${b.blockInPhase} once ${m} of your last ${w}`;
  }).join(', ') + '.';
  // Duration: test rounds plus ~45 practice rounds per practice block, ~1.2 s to answer per
  // round, plus ~12 min of setup, instructions and breaks (the 1.0 pilot: median 60 min
  // without the questionnaire)
  const nTest = test.reduce((n, b) => n + b.trials.length, 0);
  const roundMs = TIMING.iti + 1200 + TIMING.feedback;
  const minutes = Math.round(((nTest + 45 * practice.length) * roundMs) / 60000 + 12
    + (QUESTIONNAIRE.enabled ? QUESTIONNAIRE.minutes : 0));
  const F = design.keys.cat1.toUpperCase();
  const J = design.keys.cat0.toUpperCase();
  const pct = (p) => `${Math.round(p * 100)}%`;
  const tierRows = BONUS.tiers.map(([min, usd], i) => {
    const next = BONUS.tiers[i + 1];
    const range = next ? `${pct(min)} to ${pct(next[0])}` : `${pct(min)} or more`;
    return `<tr><td>${range}</td><td>${formatDollars(Math.round(usd * 100) / 100)}</td></tr>`;
  }).join('');

  const pages = [
    page(`<h2>Welcome!</h2>
      <p class="center">In each round, two symbols appear, one on each side of the centre of the screen.
      Every pair of symbols belongs to one of two groups: press ${key(F)} for one group and ${key(J)} for
      the other. At first you will have to guess. After every answer you see whether you were right, so you
      can learn which pairs go with ${key(F)} and which go with ${key(J)}.</p>
      ${exampleScreen(example, { height: 170 })}`),
    page(`<h2>Each round</h2>
      <p class="center">Each round starts with a small target in the centre: look at it. Then the two symbols
      appear: look at them as much as you like, and press ${key(F)} or ${key(J)} within
      ${TIMING.response / 1000} seconds.</p>
      <div class="example-row">
        ${exampleScreen(example, { height: 140, symbols: false })}
        ${exampleScreen(example, { height: 140 })}
      </div>`),
    page(`<h2>Feedback</h2>
      <p class="center">After you answer, circles appear around the symbols: <b style="color: ${FEEDBACK.correct}">green</b>
      if you were right, <b style="color: ${FEEDBACK.wrong}">red</b> if you were wrong or too slow.</p>
      <div class="example-row">
        ${exampleScreen(example, { height: 140, outcome: FEEDBACK.correct })}
        ${exampleScreen(example, { height: 140, outcome: FEEDBACK.wrong })}
      </div>`),
    page(`<h2>Your performance bonus</h2>
      <p class="center">On top of your base payment, you can earn a <b>bonus of up to ${formatDollars(maxBonus(test.length))}</b>,
      paid through Prolific. Each of the ${test.length} <b>test</b> blocks earns its own bonus, from the share of
      your answers in that block that are right:</p>
      <table class="tiers"><tr><th>Correct in the block</th><th>Bonus for the block</th></tr>
        <tr><td>below ${pct(BONUS.tiers[0][0])}</td><td>$0.00</td></tr>${tierRows}</table>
      <p class="center">Answers that are too slow count as wrong. Practice answers do not count, and the bonus
      never reduces your base payment. You see each block's bonus at the break after it.</p>`),
    page(`<h2>Procedure</h2>
      <ul>
      <li>First, ${practice.length} practice block${practice.length > 1 ? 's' : ''}: each side shows one of
        ${practice[0]?.size ?? 2} symbols. ${criteriaText}</li>
      <li>Then ${test.length} test blocks: each side shows one of ${test[0]?.size ?? 4} symbols,
        ${test[0]?.trials.length ?? 0} rounds each.</li>
      <li>Every block has <b>new</b> symbols, so you learn which pairs go with ${key(F)} and ${key(J)} from scratch.</li>
      <li>You can rest between blocks.</li>
      ${QUESTIONNAIRE.enabled ? '<li>At the end, a few short questions about the task.</li>' : ''}
      <li>Expected duration: about ${minutes} minutes.</li></ul>
      <p class="center small">If the quick questions below are answered wrongly twice, or the first practice
      shows that the task is not a good match (for example, many answers too slow), the study ends early and you
      receive ${formatDollars(SCREENING.payUsd)} for your time.</p>
      <p class="center">Next, a few quick questions to check the instructions.</p>`),
  ];

  const instructions = {
    type: jsPsychInstructions,
    pages,
    show_clickable_nav: true,
    allow_backward: true,
    key_forward: 'ArrowRight',
    key_backward: 'ArrowLeft',
    data: { part: 'instructions' },
  };

  const answers = {
    what_to_do: `Press ${F} or ${J}, depending on the pair`,
    feedback: 'Green circles: right; red circles: wrong or too slow',
    bonus: 'Each test block pays more the more of its answers I get right',
    look_where: 'At the target in the middle',   // between rounds
  };
  // attempts / passed are updated by the quiz itself, so the retry screen and the loop see
  // the current attempt
  let attempts = 0;
  let passed = false;
  const quiz = {
    type: jsPsychSurveyMultiChoice,
    preamble: '<h3>Quick check</h3>',
    questions: [
      { name: 'what_to_do', prompt: 'What do you do when the two symbols appear?',
        options: [answers.what_to_do, 'Press SPACE', 'Nothing, just look at them'], required: true },
      { name: 'feedback', prompt: 'How do you know whether your answer was right?',
        options: [answers.feedback, 'There is no feedback', 'The symbols disappear when I am right'], required: true },
      { name: 'bonus', prompt: 'How is your performance bonus earned?',
        options: [answers.bonus, 'Everyone gets the same bonus', 'Only the practice answers count'], required: true },
      { name: 'look_where', prompt: 'Where should you look <b>between</b> rounds, before the symbols appear?',
        options: [answers.look_where, 'At the left edge of the screen'], required: true },
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
      stimulus: () => page('<p class="center">Some answers were not right. Please read the instructions again.'
        + (onFail && attempts === COMPREHENSION.maxAttempts - 1 ? '<br>This is your last try.' : '') + '</p>'),
      choices: ['Show the instructions again'],
      data: { part: 'comprehension_retry' },
    }],
    conditional_function: () => !passed && attempts < COMPREHENSION.maxAttempts,
  };

  // After maxAttempts failures: screened out (onFail), or without screening the participant
  // continues anyway and comprehension_passed = false in the data flags them for exclusion.
  const loop = {
    timeline: [instructions, quiz, retry],
    loop_function: () => !passed && attempts < COMPREHENSION.maxAttempts,
  };
  const failed = {
    type: jsPsychCallFunction,
    func: () => { if (!passed && onFail) onFail(); },
    data: { part: 'comprehension_check' },
  };
  return { timeline: [loop, failed] };
}
