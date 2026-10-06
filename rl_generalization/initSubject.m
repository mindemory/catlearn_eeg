function subjectInfo = initSubject(parameters, subj_dir)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
%
% Per-subject assignments that must stay fixed across blocks and sessions:
% which four noise patches the subject sees, and which task set they learn.
% Drawn at random the first time a subject is run and saved to subj_dir;
% every later run of that subject loads the saved file instead.
%
% subjectInfo fields:
%   stimIdx       [a1 a2 b1 b2], indices into the stimulus pool. a1/a2 are
%                 the two levels of dimension A (shown at location A),
%                 b1/b2 the two levels of dimension B (location B).
%   stimFiles     full paths of those four PNGs, same order
%   taskSet       0, 1 or 2, numbered as in the 2x2 kernel-loading figure:
%                 0 'A'   category = level of dimension A
%                 1 'B'   category = level of dimension B
%                 2 'AB'  category = whether the levels differ (XOR)
%   taskName      'A', 'B' or 'AB'
%   categoryKeys  {key for category 0, key for category 1}
taskNames = {'A', 'B', 'AB'};
infoFile = fullfile(subj_dir, ['subjectInfo_sub' parameters.subject '.mat']);

if exist(infoFile, 'file') == 2
    loaded = load(infoFile, 'subjectInfo');
    subjectInfo = loaded.subjectInfo;
    if ~isfield(subjectInfo, 'taskSet') || numel(subjectInfo.stimIdx) ~= 4
        error(['initSubject: %s was made by an older version of the task (not 4 patches + taskSet). ' ...
            'Delete it to draw a fresh assignment.'], infoFile)
    end
    fprintf('initSubject: loaded existing assignment from %s\n', infoFile);
else
    if exist(subj_dir, 'dir') ~= 7
        mkdir(subj_dir);
    end
    subjectInfo.subject = parameters.subject;
    subjectInfo.stimIdx = randperm(parameters.nStimPool, 4);
    subjectInfo.taskSet = randi(numel(taskNames)) - 1;
    subjectInfo.taskName = taskNames{subjectInfo.taskSet + 1};
    subjectInfo.categoryKeys = parameters.resp_keys(randperm(2));
    subjectInfo.created = char(datetime('now'));
    save(infoFile, 'subjectInfo');
    fprintf('initSubject: new assignment saved to %s\n', infoFile);
end

% Resolve file paths every run, so a subject assigned on one machine can be
% run on another as long as parameters.stimDir points at the same pool.
subjectInfo.stimFiles = arrayfun(@(k) fullfile(parameters.stimDir, sprintf(parameters.stimPattern, k)), ...
    subjectInfo.stimIdx, 'UniformOutput', false);
missing = subjectInfo.stimFiles(cellfun(@(f) exist(f, 'file') ~= 2, subjectInfo.stimFiles));
if ~isempty(missing)
    error('initSubject: stimulus file(s) not found: %s. Check parameters.stimDir.', strjoin(missing, ', '))
end

fprintf('  location A: patches %02d / %02d | location B: patches %02d / %02d\n', subjectInfo.stimIdx);
fprintf('  task set %d (%s) | category 0 = %s, category 1 = %s\n', subjectInfo.taskSet, subjectInfo.taskName, ...
    upper(subjectInfo.categoryKeys{1}), upper(subjectInfo.categoryKeys{2}));
end
