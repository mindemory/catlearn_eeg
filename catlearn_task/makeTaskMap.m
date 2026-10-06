function tMap = makeTaskMap(parameters, taskset, block)
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
%
% Trial table for one block, read from the subject's offline taskset
% (makeTaskset.m) -- nothing is randomized here, so a block replays exactly
% as designed. Kept separate from the task loop so the design can change
% without touching catlearn_task.m.
blk = taskset.blocks(block);
n = numel(blk.stim);
if n ~= parameters.nTrials
    warning('makeTaskMap: taskset has %d trials in block %d, parameters.nTrials is %d; using the taskset.', ...
        n, block, parameters.nTrials);
end

tMap.nTrials = n;
tMap.rule = blk.rule;
tMap.fractals = blk.fractals;     % this block's [a1 a2 b1 b2] file numbers
tMap.stim = blk.stim;             % compound 1-4 (= 2*levelA + levelB + 1)
tMap.levelA = blk.levelA;         % 0 = a1, 1 = a2
tMap.levelB = blk.levelB;         % 0 = b1, 1 = b2
tMap.category = blk.category;     % 0/1 under this block's rule
tMap.correctKey = blk.correctKey; % 'f' / 'j'
tMap.iti = blk.iti;               % s, jittered within parameters.itiDuration
end
