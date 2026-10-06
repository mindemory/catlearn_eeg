function tMap = makeTaskMap(parameters, subjectInfo)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
%
% Builds the trial table for one block of the 2x2. Each trial shows one of
% the two dimension-A patches at location A and one of the two dimension-B
% patches at location B, giving four compound stimuli. Each compound
% appears nTrials/4 times, in random order. The subject's task set maps the
% compound to a category, and their key mapping maps the category to F/J.
n = parameters.nTrials;
if mod(n, 4) ~= 0
    error('makeTaskMap: nTrials (%d) must be a multiple of 4 to balance the four compounds.', n)
end

% All four compounds, [levelA levelB]
compounds = [1 1; 1 2; 2 1; 2 2];
order = repmat(1:4, 1, n/4);
order = order(randperm(n));

tMap.nTrials = n;
tMap.compound = order;                   % 1-4, row of compounds above
tMap.levelA = compounds(order, 1)';      % dimension-A level at location A (1 or 2)
tMap.levelB = compounds(order, 2)';      % dimension-B level at location B (1 or 2)

switch subjectInfo.taskSet
    case 0 % A: category follows dimension A
        category = tMap.levelA == 2;
    case 1 % B: category follows dimension B
        category = tMap.levelB == 2;
    case 2 % AB: XOR of the two dimensions
        category = tMap.levelA ~= tMap.levelB;
    otherwise
        error('makeTaskMap: unknown taskSet %d.', subjectInfo.taskSet)
end
tMap.category = double(category);                              % 0 or 1
tMap.correctKey = subjectInfo.categoryKeys(tMap.category + 1); % 'f' or 'j' per trial
end
