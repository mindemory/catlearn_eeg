// Consent, instructions and the comprehension check. Text only here; the trial screens
// are in display.js and the task timeline in task.js.

import { COMPREHENSION, CONTACT, FEEDBACK, PRACTICE_CRITERION, REWARD, TEST_CRITERION, TIMING } from './config.js';
import { coinBar, exampleMachine } from './display.js';
import { formatCoins } from './reward.js';

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
 * @param {object} example      design.example: symbols of a practice and a test machine
 */
export function instructionsWithCheck(design, example) {
  const practice = design.blocks.filter((b) => b.phase === 'practice');
  const test = design.blocks.filter((b) => b.phase === 'test');
  const { window: WINDOW, minCorrect: MIN_CORRECT } = PRACTICE_CRITERION;
  // Duration at most: every test round plus ~30 practice rounds per practice block, ~1 s of
  // waiting and ~1.2 s to answer per round, plus ~8 min of setup and instructions
  const nTest = test.reduce((n, b) => n + b.trials.length, 0);
  const roundMs = 1000 + TIMING.pull + 1200 + TIMING.feedback;
  const minutes = Math.round(((nTest + 30 * practice.length) * roundMs) / 60000 + 8);
  const coins = (n) => `${formatCoins(n)} coin${Math.abs(n) === 1 ? '' : 's'}`;
  // Coin examples from the actual design and payoffs
  const nPaid = design.blocks.filter((b) => REWARD.bonusPhases.includes(b.phase)).reduce((n, b) => n + b.trials.length, 0);
  const coinsAt = (p) => Math.round(nPaid * (p * REWARD.correct + (1 - p) * REWARD.wrong));
  const Y = design.keys.yes.toUpperCase();
  const N = design.keys.no.toUpperCase();
  const wrongCost = coins(Math.abs(REWARD.wrong));
  const { practice: exPractice, test: exTest } = example;
  const waitS = TIMING.wait / 1000;

  const pages = [
    page(`<h2>Welcome to the slot machines!</h2>
      <p class="center">Each machine shows symbols around the centre of the screen. Some combinations of
      symbols <b>win</b>, the others <b>lose</b>. Your job is to learn which is which.</p>
      <div class="example-row">
        ${exampleMachine(exPractice, { height: 190 })}
        ${exampleMachine(exTest, { height: 190 })}
      </div>
      <p class="center small">Practice machines show two symbols; test machines show three patterned patches.</p>`),
    page(`<h2>Each round</h2>
      <p class="center">Press ${key('space')} to pull the lever (if you wait ${waitS} seconds, the machine starts
      by itself). The symbols appear. Answer: <b>will these symbols win?</b><br>
      Press ${key(Y)} for <b>YES</b> or ${key(N)} for <b>NO</b>, within ${TIMING.response / 1000} seconds.</p>
      <div class="example-row">
        ${exampleMachine(exPractice, { height: 190, symbols: false, hint: 'press SPACE to play' })}
        ${exampleMachine(exPractice, { height: 190 })}
      </div>`),
    page(`<h2>See the payout</h2>
      <p class="center">Circles then appear around the symbols: <b style="color: ${FEEDBACK.correct}">green</b>
      if you were right (<b>+${REWARD.correct} coins</b>), <b style="color: ${FEEDBACK.wrong}">red</b> if you were
      wrong or too slow (<b>&minus;${wrongCost}</b>).</p>
      <div class="example-row">
        ${exampleMachine(exPractice, { height: 190, outcome: FEEDBACK.correct })}
        ${exampleMachine(exPractice, { height: 190, outcome: FEEDBACK.wrong })}
      </div>`),
    page(`<h2>Your coins</h2>
      <p class="center">A bar at the top of the screen shows your coins: it fills as you win and shrinks as
      you lose.</p>
      <div class="example-bar">${coinBar({ coins: coinsAt(0.8) / 2, max: nPaid * REWARD.correct, paid: true,
                                            label: `${formatCoins(Math.round(coinsAt(0.8) / 2))} coins` })}</div>
      <p class="center">Coins you win in the <b>test</b> are paid to you as a bonus on top of your
      base payment (${formatCoins(REWARD.coinsPerDollar)} coins = $1). Practice coins are not paid.
      For example, predicting 80% of the test rounds correctly earns about ${formatCoins(coinsAt(0.8))} coins;
      a perfect score earns ${formatCoins(coinsAt(1))}. If you master a test machine early, its rounds left
      are paid as if you got them all right.</p>`),
    page(`<h2>Keep your eyes on the centre</h2>
      <p class="center">Please always keep your eyes on the small black-and-white target in the middle of the
      screen, even when the symbols appear.</p>
      ${exampleMachine(exTest, { height: 240 })}`),
    page(`<h2>Procedure</h2>
      <ul>
      <li>First, ${practice.length} practice machine${practice.length > 1 ? 's' : ''} with two symbols (left and
        right). Each ends once ${MIN_CORRECT} of your last ${WINDOW} answers are correct.</li>
      <li>Then ${test.length} test machine${test.length > 1 ? 's' : ''} with three patterned patches (left, right
        and top), up to ${test[0]?.trials.length ?? 0} rounds each. Each stops early once
        ${TEST_CRITERION.minCorrect} of your last ${TEST_CRITERION.window} answers are correct.</li>
      <li>Every machine has <b>new</b> symbols, so you learn its winning combinations from scratch.</li>
      <li>You can rest between machines.</li>
      <li>Duration: at most about ${minutes} minutes.</li></ul>
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
    yes_key: Y,
    start_round: `Press SPACE (or wait ${waitS} seconds)`,
    wrong_cost: `You lose ${wrongCost}`,
    paid_coins: 'Coins from the test',
    look_where: 'At the target in the middle',
  };
  // attempts / passed are updated by the quiz itself, so the retry screen and the loop see
  // the current attempt
  let attempts = 0;
  let passed = false;
  const quiz = {
    type: jsPsychSurveyMultiChoice,
    preamble: '<h3>Quick check</h3>',
    questions: [
      { name: 'yes_key', prompt: 'Which key do you press if you think the symbols <b>will win</b>?',
        options: ['F', 'J'], required: true },
      { name: 'start_round', prompt: 'How does each round start?',
        options: [`Press SPACE (or wait ${waitS} seconds)`, 'Press F or J'], required: true },
      { name: 'wrong_cost', prompt: 'What happens when your prediction is wrong?',
        options: [`You lose ${wrongCost}`, 'Nothing happens', `You win ${coins(REWARD.correct)}`], required: true },
      { name: 'paid_coins', prompt: 'Which coins are paid to you as a bonus?',
        options: ['Coins from the test', 'Coins from the practice', 'No coins are paid'], required: true },
      { name: 'look_where', prompt: 'Where should you keep your eyes?',
        options: ['At the target in the middle', 'At whichever symbol looks most important'], required: true },
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
