function taskset = makeTaskset(subjIDs, overwrite)
% taskset = makeTaskset(subjIDs, overwrite)
%
% Created by Mrugank Dake, Dartmouth College (09/30/2026)
%
% Builds each subject's taskset OFFLINE, before the session, and saves it to
% parameters.tasksetDir as taskset_subNN.mat. catlearn_task loads it and
% errors if it is missing, so every design choice is fixed (and inspectable)
% before the subject arrives. Needs no Psychtoolbox.
%
% Per subject:
%   - 4 NEW fractals every block, [a1 a2 b1 b2], all 4 x nBlocks drawn at random
%     from the 72 without replacement, so no fractal appears in two blocks.
%     taskset.fractals is (nBlocks, 4); taskset.blocks(b).fractals is block b's row.
%   - Counterbalanced by subject ID (24 cells, cycling with ID, 3 sequences
%     varying fastest):
%       rule sequence   A-A-AB | A-B-AB | AB-AB-A   (parameters.ruleSequences)
%       side of A       left | right  (B on the other side)
%       tagging freq    left 12 / right 15 Hz | left 15 / right 12 Hz
%       keys            category 0 -> F, 1 -> J | 0 -> J, 1 -> F
%   - Per block: trial order (each compound nTrials/4 times, shuffled with at
%     most 3 identical compounds and 4 identical correct keys in a row) and
%     ITI jitter.
%
% Everything random comes from rng(1000 + subject number, 'twister'), so the
% same ID always gives the same taskset, on any machine.
%
% Compound index s = 2*levelA + levelB + 1 (1-4): A shows a1 (levelA 0) or a2
% (levelA 1), B shows b1 or b2. Category under each rule (0/1):
%   A: levelA    B: levelB    AB: xor(levelA, levelB)
%
% Examples:
%   makeTaskset(1)              % subject 01
%   makeTaskset(1:24)           % one full counterbalancing cycle
%   makeTaskset(3, true)        % regenerate subject 03, overwriting the file

if nargin < 2 || isempty(overwrite); overwrite = false; end

for subj = subjIDs(:)'
    taskset = makeOne(subj);
    parameters = loadParameters(taskset.subject);
    if exist(parameters.tasksetDir, 'dir') ~= 7
        mkdir(parameters.tasksetDir);
    end
    file = fullfile(parameters.tasksetDir, ['taskset_sub' taskset.subject '.mat']);
    if exist(file, 'file') == 2 && ~overwrite
        fprintf('makeTaskset: %s exists, kept (pass overwrite = true to regenerate)\n', file);
        continue;
    end
    save(file, 'taskset');
    fracStr = strjoin(arrayfun(@(b) sprintf('B%d [%s]', b, num2str(taskset.fractals(b, :))), ...
        1:size(taskset.fractals, 1), 'UniformOutput', false), ' ');
    fprintf('sub%s: rules %s | A on the %s | %d Hz left, %d Hz right | category 0 = %s, 1 = %s | fractals a1 a2 b1 b2: %s\n', ...
        taskset.subject, strjoin(taskset.rules, '-'), taskset.sideA, taskset.freqLeftRight, ...
        upper(taskset.categoryKeys{1}), upper(taskset.categoryKeys{2}), fracStr);
end
end

function taskset = makeOne(subj)
if ischar(subj) || isstring(subj); subj = str2double(subj); end
if ~(isscalar(subj) && subj >= 1 && subj == round(subj))
    error('makeTaskset: subject IDs must be positive integers.')
end
subjID = num2str(subj, '%02d');
parameters = loadParameters(subjID);
n = parameters.nTrials;
if mod(n, 4) ~= 0
    error('makeTaskset: parameters.nTrials (%d) must be a multiple of 4.', n)
end
seed = 1000 + subj;
rng(seed, 'twister');

% Counterbalancing cell
nSeq = numel(parameters.ruleSequences);
cell0 = mod(subj - 1, nSeq * 8);
seqIdx = mod(cell0, nSeq) + 1;
sideIdx = mod(floor(cell0 / nSeq), 2);
freqIdx = mod(floor(cell0 / (nSeq * 2)), 2);
keyIdx = mod(floor(cell0 / (nSeq * 4)), 2);

taskset.subject = subjID;
taskset.seed = seed;
taskset.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
taskset.counterbalanceCell = cell0 + 1;
if 4 * parameters.nBlocks > parameters.nFractals
    error('makeTaskset: %d blocks x 4 fractals exceeds the %d available.', parameters.nBlocks, parameters.nFractals)
end
% File numbers, one row per block: [a1 a2 b1 b2]
taskset.fractals = reshape(randperm(parameters.nFractals, 4 * parameters.nBlocks), 4, parameters.nBlocks)';
taskset.sequenceIdx = seqIdx;
taskset.rules = parameters.ruleSequences{seqIdx};
sides = {'left', 'right'};
taskset.sideA = sides{sideIdx + 1};
taskset.freqLeftRight = parameters.tagFreqs(1 + mod([0 1] + freqIdx, 2)); % [left right], Hz
if keyIdx == 0
    taskset.categoryKeys = {'f', 'j'}; % key for category 0, category 1
else
    taskset.categoryKeys = {'j', 'f'};
end

% Compound table (the same in every block)
s = (1:4)';
levelA = floor((s - 1) / 2);
levelB = mod(s - 1, 2);
taskset.compounds = table(s, levelA, levelB);

for b = 1:parameters.nBlocks
    rule = taskset.rules{b};
    switch rule
        case 'A';  category = levelA;
        case 'B';  category = levelB;
        case 'AB'; category = double(xor(levelA, levelB));
        otherwise; error('makeTaskset: unknown rule %s', rule)
    end
    % Balanced order with no long runs of one compound or one correct key
    for attempt = 1:10000
        stim = repmat(s, n / 4, 1);
        stim = stim(randperm(n));
        if maxRun(stim) <= 3 && maxRun(category(stim)) <= 4
            break;
        end
    end
    blk.block = b;
    blk.rule = rule;
    blk.fractals = taskset.fractals(b, :);              % [a1 a2 b1 b2] file numbers
    blk.categoryOfCompound = category';                 % category of s = 1..4
    blk.stim = stim';
    blk.levelA = levelA(stim)';
    blk.levelB = levelB(stim)';
    blk.category = category(stim)';
    blk.correctKey = taskset.categoryKeys(category(stim)' + 1);
    blk.iti = parameters.itiDuration(1) + rand(1, n) * diff(parameters.itiDuration);
    taskset.blocks(b) = blk;
end
end

function r = maxRun(x)
% Longest run of identical consecutive values
x = x(:);
edges = [true; diff(x) ~= 0; true];
r = max(diff(find(edges)));
end
