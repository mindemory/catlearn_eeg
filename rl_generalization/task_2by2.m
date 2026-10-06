function task_2by2(subjID, config, block, debugMode)
% task_2by2(subjID, config, block, debugMode)
%
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
%
% 2x2 RL generalization task. Location A shows one of two patches
% (dimension A) and location B one of two others (dimension B), giving four
% compound stimuli; the locations come from the spatial configuration. The
% subject sorts each compound into a category with F or J and gets feedback.
%
% Trial structure:
%   fixation   1.0 s        white fixation cross alone
%   stimuli    <= 4.0 s     both patches + fixation, until F/J or the deadline
%   feedback   1.5 s        'Correct' / 'Incorrect' / 'Too slow'
% There is no separate ITI: feedback is followed directly by the next
% trial's fixation.
%
% Each subject gets four patches, one of the three task sets (0 = A,
% 1 = B, 2 = AB/XOR, as in the kernel-loading figure) and a category->key
% mapping, all drawn at random on their first run and reused after that
% (see initSubject.m). Each block shows each compound nTrials/4 times in
% random order (makeTaskMap.m).
%
% Data is saved under parameters.dataDir (~/Documents/data/catlearn_eeg/
% rl_generalization/subNN), outside the code repo.
%
% Inputs:
%   subjID    numeric or string subject id (zero-padded to 2 digits)
%   config    spatial configuration 1-18 (see spatialConfig.m),
%             default 2 = horizontal, mid (left/right of fixation)
%   block     block number, default 1
%   debugMode 1 = transparent window + SkipSyncTests, keyboard not captured
%             0 = full-screen, timing enforced (default)
%
% Press ESCAPE during any stimulus window to abort; data collected so far
% is still saved.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 1 || isempty(subjID); error('task_2by2: subjID is required.'); end
if nargin < 2 || isempty(config); config = 2; end
if nargin < 3 || isempty(block); block = 1; end
if nargin < 4 || isempty(debugMode); debugMode = 0; end

close all; clc;
subjID = num2str(subjID, '%02d'); % convert subjID to string

% Resolve paths relative to this file, so the task runs from any cwd
taskDir = fileparts(mfilename('fullpath'));
addpath(taskDir);

% Validate config before any hardware is touched
spatialConfig(config);

% Add Psychtoolbox if it is not already on the path. The first entry is the
% inner toolbox folder of a git clone of Psychtoolbox-3; the repo root also
% holds C/Python sources that must not go on the MATLAB path.
if exist('Screen', 'file') ~= 3 && exist('Screen', 'file') ~= 2
    ptbCandidates = {fullfile(getenv('HOME'), 'Documents', 'MATLAB', 'Psychtoolbox', 'Psychtoolbox'), ...
                     '/Applications/Psychtoolbox', ...
                     '/usr/share/psychtoolbox-3'};
    for ii = 1:numel(ptbCandidates)
        if exist(ptbCandidates{ii}, 'dir') == 7
            addpath(genpath(ptbCandidates{ii}));
            break;
        end
    end
end
if exist('Screen', 'file') == 0
    error('task_2by2: Psychtoolbox not found on the path.')
end

% Load settings
parameters = loadParameters(subjID);
parameters.config = config;
parameters.isDemoMode = logical(debugMode);
parameters.debugMode = logical(debugMode);

if exist(parameters.stimDir, 'dir') ~= 7
    error('task_2by2: stimulus folder not found: %s. Set parameters.stimDir in loadParameters.m.', parameters.stimDir)
end

subj_dir = fullfile(parameters.dataDir, ['sub' subjID]);

rng('shuffle');
parameters.rngState = rng; % lets the block's random draws (trial order, first-run assignment) be reproduced

if debugMode
    Screen('Preference', 'SkipSyncTests', 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Open the screen and peripherals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Everything past this point runs inside try/catch so a crash cannot leave a
% black full-screen window and a captured keyboard behind.
kbx = [];
try
    screen = initScreen(parameters);
    [kbx, parameters] = initPeripherals(parameters);
    subjectInfo = initSubject(parameters, subj_dir);
    stim = initStimuli(parameters, screen, subjectInfo, config);
    parameters = initFiles(parameters, screen, subj_dir, kbx, block);
    tMap = makeTaskMap(parameters, subjectInfo);
    nTrials = tMap.nTrials;
    fprintf('config %d: %s\n', config, stim.configName);

    if ~debugMode
        HideCursor(screen.id);
        ListenChar(-1); % stop keypresses leaking into the MATLAB console
    end

    timeReport = struct; % flip timestamps, one entry per trial
    respReport = struct; % subject responses

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Instructions and block start
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    KbQueueStart(kbx);
    KbQueueFlush(kbx);
    showprompts(screen, 'WelcomeWindow');
    waitForKey(kbx, parameters.space_key);

    showprompts(screen, 'BlockStart', block);
    WaitSecs(2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Task starts
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    aborted = false;
    lastTrial = 0;
    nextFixOnset = 0; % 0 == flip at the next retrace
    exitCode = KbName(parameters.exit_key);
    respCodes = KbName(parameters.resp_keys);

    for trial = 1:nTrials
        fprintf('block %02d, trial %02d ... ', block, trial);

        %------------------------------------------------------------------
        % Fixation window (1 s, white cross)
        %------------------------------------------------------------------
        drawTextures(parameters, screen, stim, 'FixationCross');
        tFixOn = Screen('Flip', screen.win, nextFixOnset);
        timeReport(trial).fixOnset = tFixOn;
        if trial > 1 % this flip is what ends the previous trial's feedback
            timeReport(trial-1).feedbackDuration = tFixOn - timeReport(trial-1).feedbackOnset;
        end

        %------------------------------------------------------------------
        % Stimulus window (until response, max 4 s; fixation stays up)
        %------------------------------------------------------------------
        drawTextures(parameters, screen, stim, 'Stimuli', [tMap.levelA(trial) tMap.levelB(trial)]);
        drawTextures(parameters, screen, stim, 'FixationCross');
        tStimOn = Screen('Flip', screen.win, tFixOn + parameters.fixDuration - screen.ifi/2);
        timeReport(trial).stimOnset = tStimOn;
        timeReport(trial).fixDuration = tStimOn - tFixOn;

        % Count responses from stimulus onset onwards
        KbQueueFlush(kbx);

        % Poll until half a frame before the deadline, so a timeout's
        % feedback flip lands on the 4 s retrace rather than one frame late
        respKey = '';
        respTime = NaN;
        respDeadline = tStimOn + parameters.stimDuration - screen.ifi/2;
        while GetSecs < respDeadline
            % firstPress holds a GetSecs timestamp per keycode. Check the
            % response keys as a group rather than taking the earliest of
            % ALL pressed keys, so a simultaneous non-response key can't
            % discard the real response.
            [pressed, firstPress] = KbQueueCheck(kbx);
            if pressed
                if firstPress(exitCode) > 0
                    aborted = true;
                    break;
                end
                pressTimes = firstPress(respCodes);
                hit = find(pressTimes > 0);
                if ~isempty(hit)
                    [pressTime, iEarliest] = min(pressTimes(hit));
                    respKey = parameters.resp_keys{hit(iEarliest)};
                    respTime = pressTime - tStimOn;
                    break;
                end
            end
        end
        if aborted
            fprintf('aborted by experimenter.\n');
            break;
        end

        timedOut = isempty(respKey);
        correct = ~timedOut && strcmp(respKey, tMap.correctKey{trial});

        respReport(trial).key = respKey;
        respReport(trial).correctKey = tMap.correctKey{trial};
        respReport(trial).RT = respTime; % seconds from stimulus onset
        respReport(trial).correct = correct;
        respReport(trial).timedOut = timedOut;
        respReport(trial).compound = tMap.compound(trial);
        respReport(trial).levelA = tMap.levelA(trial);
        respReport(trial).levelB = tMap.levelB(trial);
        respReport(trial).category = tMap.category(trial);
        lastTrial = trial; % only counted once the trial's data is on record

        %------------------------------------------------------------------
        % Feedback window (1.5 s), replaces the stimuli right away
        %------------------------------------------------------------------
        if timedOut
            showprompts(screen, 'Timeout', [], 0, parameters.timeoutColor);
        elseif correct
            showprompts(screen, 'Correct', [], 0, parameters.correctColor);
        else
            showprompts(screen, 'Incorrect', [], 0, parameters.incorrectColor);
        end
        tFbOn = Screen('Flip', screen.win);
        timeReport(trial).feedbackOnset = tFbOn;
        timeReport(trial).stimDuration = tFbOn - tStimOn;

        % Feedback ends when the next trial's fixation flips
        nextFixOnset = tFbOn + parameters.feedbackDuration - screen.ifi/2;

        if timedOut
            fprintf('no response\n');
        else
            verdict = {'incorrect', 'correct'};
            fprintf('%s, %s, RT = %.3f s\n', upper(respKey), verdict{correct+1}, respTime);
        end
    end

    % Hold the last trial's feedback for its full duration before the
    % end-of-block screen replaces it
    if ~aborted && lastTrial > 0
        WaitSecs('UntilTime', nextFixOnset);
        timeReport(lastTrial).feedbackDuration = GetSecs - timeReport(lastTrial).feedbackOnset;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save data and close
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if lastTrial > 0 && isfield(respReport, 'correct')
        accuracy = 100 * mean([respReport(1:lastTrial).correct]);
        medRT = median([respReport(1:lastTrial).RT], 'omitnan');
        nTimeout = sum([respReport(1:lastTrial).timedOut]);
    else
        respReport = struct([]); % no completed trials, keep the file honest
        accuracy = NaN; medRT = NaN; nTimeout = 0;
    end
    fprintf('\nBlock %02d done: %d/%d trials, accuracy %.1f%%, median RT %.3f s, %d timeouts\n', ...
        block, lastTrial, nTrials, accuracy, medRT, nTimeout);

    matFile.parameters = parameters;
    matFile.screen = screen;
    matFile.stim = rmfield(stim, 'tex'); % texture handles are meaningless once closed
    matFile.subjectInfo = subjectInfo;
    matFile.tMap = tMap;
    matFile.timeReport = timeReport;
    matFile.respReport = respReport;
    matFile.aborted = aborted;
    matFile.nTrialsCompleted = lastTrial;
    save([parameters.block_dir filesep parameters.matFile], 'matFile')
    fprintf('Saved to %s\n', parameters.block_dir);

    % End-of-block screen
    if ~aborted
        showprompts(screen, 'BlockEnd', accuracy);
        waitForKey(kbx, {parameters.space_key, parameters.exit_key});
    end
    showprompts(screen, 'EndExperiment');
    WaitSecs(2);

    cleanUp(kbx, stim, debugMode);

catch ME
    if exist('stim', 'var')
        cleanUp(kbx, stim, debugMode);
    else
        cleanUp(kbx, [], debugMode);
    end
    rethrow(ME);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local helpers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pressedKey = waitForKey(kbx, acceptKeys)
% Blocks until one of acceptKeys is pressed; whatever is on screen stays up.
if ischar(acceptKeys); acceptKeys = {acceptKeys}; end
acceptCodes = KbName(acceptKeys);
pressedKey = '';
KbQueueFlush(kbx);
while isempty(pressedKey)
    [pressed, firstPress] = KbQueueCheck(kbx);
    if pressed
        pressTimes = firstPress(acceptCodes);
        hit = find(pressTimes > 0);
        if ~isempty(hit)
            [~, iEarliest] = min(pressTimes(hit));
            pressedKey = acceptKeys{hit(iEarliest)};
        end
    end
    WaitSecs(0.01); % idle screen, no need to spin a core
end
KbQueueFlush(kbx);
end

function cleanUp(kbx, stim, debugMode)
% Restore the machine to a usable state, in an order that is safe even if
% initialization only got partway through.
if ~debugMode
    ListenChar(1);
    ShowCursor;
end
if ~isempty(kbx)
    try KbQueueStop(kbx); KbQueueRelease(kbx); catch; end %#ok<CTCH>
end
if ~isempty(stim) && isstruct(stim) && isfield(stim, 'tex')
    try Screen('Close', stim.tex); catch; end %#ok<CTCH>
end
sca;
Priority(0);
if debugMode
    PsychDebugWindowConfiguration(0, 1); % undo the transparent-window setting
end
end
