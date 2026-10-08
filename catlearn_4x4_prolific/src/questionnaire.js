// Post-task questionnaire, after the last test block and before the bonus screen (~4 min).
// Pages: difficulty per block; strategy per block and the hardest block; effort (NASA-TLX items);
// engagement; individual differences (Need for Cognition short form, handedness, gaming).
// Every answer is saved on its page's row (part = 'questionnaire', page = ...) as one column
// per question, q_<name>; Likert answers are 1-based (1 = the left label).

import { STIMULI } from './config.js';

const page = (html) => `<div class="page">${html}</div>`;

// Likert answers are 0-based in jsPsych; store 1-based columns
function flatten(data, likert = false) {
  Object.entries(data.response ?? {}).forEach(([k, v]) => {
    data[`q_${k}`] = likert && typeof v === 'number' ? v + 1 : v;
  });
}

const scale7 = (low, high) => [low, '', '', '', '', '', high];

/**
 * @param {object} design  from buildDesign (number of test blocks)
 */
export function questionnaireTimeline(design) {
  const nTest = design.blocks.filter((b) => b.phase === 'test').length;
  const ordinal = ['first', 'second', 'third', 'fourth', 'fifth'];
  const blockNames = Array.from({ length: nTest }, (_, i) => ordinal[i] ?? `${i + 1}th`);
  const difficulty = scale7('Very easy', 'Very hard');

  const intro = {
    type: jsPsychHtmlButtonResponse,
    stimulus: page(`<h2>A few questions</h2>
      <p class="center">The task is over. Before you see your bonus, please answer a few short questions
      about it (about 4 minutes). There are no right or wrong answers, and your answers do not change your
      bonus.</p>`),
    choices: ['Continue'],
    data: { part: 'questionnaire', page: 'intro' },
  };

  const difficultyPage = {
    type: jsPsychSurveyLikert,
    preamble: '<h3>How difficult was it?</h3><p>"Main blocks" are the three blocks after the practice.</p>',
    questions: [
      { name: 'difficulty_overall', prompt: 'Overall, how difficult was the task?', labels: difficulty, required: true },
      ...blockNames.map((n, i) => ({ name: `difficulty_block${i + 1}`, prompt: `The <b>${n}</b> main block`,
                                     labels: difficulty, required: true })),
    ],
    scale_width: 560,
    data: { part: 'questionnaire', page: 'difficulty' },
    on_finish: (data) => flatten(data, true),
  };

  // Strategy per block, one option per kind of rule a participant could be using (the kernel
  // model's modes): one symbol alone (A or B), each symbol separately (A + B), specific pairs
  // (AB), or guessing. Saved as q_strategy_block<k> = the option's code.
  const strategies = [
    ['left', 'Mostly the <b>left</b> symbol: some left symbols meant F, others J'],
    ['right', 'Mostly the <b>right</b> symbol: some right symbols meant F, others J'],
    ['separate', 'Each symbol <b>separately</b>: I combined what the left and the right symbol each suggested'],
    ['pairs', 'Specific <b>pairs</b>: I remembered which exact pairs went with F and which with J'],
    ['guess', 'I mostly <b>guessed</b>'],
  ];
  const testBlocks = design.blocks.filter((b) => b.phase === 'test');
  // the block's symbols, in a labelled box per side, as a reminder of which block is meant
  const thumbs = (b) => `<span class="q-sides">${[['A', 'Left symbols'], ['B', 'Right symbols']]
    .map(([side, label]) => `<span class="q-thumbs"><span class="q-side-label">${label}</span>${b.fractals[side]
      .map((f) => `<img src="${STIMULI.fractalDir}${f}.png" alt="">`).join('')}</span>`).join('')}</span>`;
  const strategyChoice = {
    type: jsPsychSurveyMultiChoice,
    preamble: `<h3>How did you decide?</h3><p>Each main block had its own symbols, shown again below.</p>`,
    questions: [
      ...testBlocks.map((b, i) => ({ name: `strategy_block${i + 1}`,
                                     prompt: `The <b>${blockNames[i]}</b> main block<br>${thumbs(b)}`,
                                     options: strategies.map(([, text]) => text), required: true })),
      { name: 'hardest_block', prompt: 'Which main block was the hardest?',
        options: [...blockNames.map((n) => `The ${n}`), 'All about the same'], required: true, horizontal: true },
    ],
    data: { part: 'questionnaire', page: 'strategy_choice' },
    on_finish: (data) => {
      flatten(data);
      testBlocks.forEach((_, i) => {
        const k = `q_strategy_block${i + 1}`;
        data[k] = strategies.find(([, text]) => text === data[k])?.[0] ?? data[k];
      });
    },
  };

  const strategyText = {
    type: jsPsychSurveyText,
    preamble: '<h3>Your strategy</h3>',
    questions: [
      { name: 'strategy', prompt: 'How did you decide whether to press F or J? Please describe any strategy you used.',
        rows: 4, columns: 70, required: true },
      { name: 'rule', prompt: 'If you found a rule (or part of one) in any block, please describe it. Leave empty if not.',
        rows: 3, columns: 70, required: false },
    ],
    data: { part: 'questionnaire', page: 'strategy_text' },
    on_finish: (data) => flatten(data),
  };

  // NASA Task Load Index items (Hart & Staveland, 1988), 7-point version
  const effort = {
    type: jsPsychSurveyLikert,
    preamble: '<h3>How demanding was it?</h3>',
    questions: [
      { name: 'tlx_mental', prompt: 'How <b>mentally</b> demanding was the task?', labels: scale7('Very low', 'Very high'), required: true },
      { name: 'tlx_physical', prompt: 'How <b>physically</b> demanding was the task?', labels: scale7('Very low', 'Very high'), required: true },
      { name: 'tlx_temporal', prompt: 'How hurried or rushed was the pace?', labels: scale7('Very low', 'Very high'), required: true },
      { name: 'tlx_performance', prompt: 'How successful were you in doing the task?', labels: scale7('Not at all', 'Completely'), required: true },
      { name: 'tlx_effort', prompt: 'How hard did you have to work to do as well as you did?', labels: scale7('Very little', 'Very hard'), required: true },
      { name: 'tlx_frustration', prompt: 'How irritated, stressed or annoyed were you?', labels: scale7('Not at all', 'Very much'), required: true },
    ],
    scale_width: 560,
    data: { part: 'questionnaire', page: 'effort' },
    on_finish: (data) => flatten(data, true),
  };

  const engagementScale = {
    type: jsPsychSurveyLikert,
    preamble: '<h3>Your attention</h3>',
    questions: [
      { name: 'focus', prompt: 'How focused were you during the main blocks?', labels: scale7('Not at all', 'Completely'), required: true },
      { name: 'motivation', prompt: 'How much did the bonus motivate you to do well?', labels: scale7('Not at all', 'Very much'), required: true },
    ],
    scale_width: 560,
    data: { part: 'questionnaire', page: 'engagement' },
    on_finish: (data) => flatten(data, true),
  };

  const engagementChoice = {
    type: jsPsychSurveyMultiChoice,
    preamble: '<p>Honest answers help us, and they do not change your payment.</p>',
    questions: [
      { name: 'distracted', prompt: 'Were you interrupted or distracted during the task?',
        options: ['No', 'Once or twice, briefly', 'Several times'], required: true, horizontal: true },
      { name: 'notes', prompt: 'Did you write anything down, or use notes, to remember the pairs?',
        options: ['No', 'Yes'], required: true, horizontal: true },
      { name: 'handedness', prompt: 'Which hand do you write with?',
        options: ['Right', 'Left', 'Both'], required: true, horizontal: true },
      { name: 'gaming', prompt: 'About how many hours a week do you play video games?',
        options: ['None', 'Less than 1', '1 to 5', '5 to 10', 'More than 10'], required: true, horizontal: true },
    ],
    data: { part: 'questionnaire', page: 'engagement_choice' },
    on_finish: (data) => flatten(data),
  };

  // Need for Cognition, 6-item short form (NCS-6; Lins de Holanda Coelho, Hanel & Wolf, 2020);
  // items 3 and 4 are reverse-scored in the analysis (marked in the item name)
  const nfcLabels = ['Extremely uncharacteristic of me', '', 'Neither', '', 'Extremely characteristic of me'];
  const nfc = {
    type: jsPsychSurveyLikert,
    preamble: '<h3>About you</h3><p>How well does each statement describe you?</p>',
    questions: [
      ['nfc1', 'I would prefer complex to simple problems.'],
      ['nfc2', 'I like to have the responsibility of handling a situation that requires a lot of thinking.'],
      ['nfc3_rev', 'Thinking is not my idea of fun.'],
      ['nfc4_rev', 'I would rather do something that requires little thought than something that is sure to challenge my thinking abilities.'],
      ['nfc5', 'I really enjoy a task that involves coming up with new solutions to problems.'],
      ['nfc6', 'I would prefer a task that is intellectual, difficult, and important to one that is somewhat important but does not require much thought.'],
    ].map(([name, prompt]) => ({ name, prompt, labels: nfcLabels, required: true })),
    scale_width: 560,
    data: { part: 'questionnaire', page: 'nfc' },
    on_finish: (data) => flatten(data, true),
  };

  const comments = {
    type: jsPsychSurveyText,
    questions: [
      { name: 'comments', prompt: 'Any problems, or anything else you would like to tell us? (optional)',
        rows: 3, columns: 70, required: false },
    ],
    data: { part: 'questionnaire', page: 'comments' },
    on_finish: (data) => flatten(data),
  };

  return { timeline: [intro, difficultyPage, strategyChoice, strategyText, effort, engagementScale, engagementChoice,
                      nfc, comments] };
}
