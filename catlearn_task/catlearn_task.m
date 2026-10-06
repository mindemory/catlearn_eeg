function catlearn_task(subjID, startBlock, debugMode)
% catlearn_task(subjID, startBlock, debugMode)
%
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
%
% Category-learning EEG task, 2x2 design with frequency-tagged fractals.
%
% Two fractals flank fixation at 6 dva, one at location A and one at
% location B. Location A shows one of two fractals (a1/a2), location B one of
% two others (b1/b2), giving 4 compounds. Every block brings 4 new fractals. The category (F or J) depends on the
% block's rule: A (which A fractal), B (which B fractal) or AB (A xor B). Each
% subject does 3 blocks with one rule sequence: A-A-AB, A-B-AB or AB-AB-A.
% Subjects are not told the rule; they learn it from feedback.
%
% Each fractal flickers (sinusoidal contrast) at its own frequency, 12 or
% 15 Hz, so the SSVEP at each frequency indexes attention to that location.
%
% Trial structure:
%   fixation      1.0 s   fixation cross alone
%   stimulus      2.0 s   both fractals flicker, cross stays up; whole cycles
%                         of both frequencies, frame-locked
%   response      <= 2.0 s from stimulus OFFSET; F or J. Keys pressed during
%                         the flicker do not count (logged as earlyPress).
%   feedback      0.5 s   'Correct' / 'Incorrect' / 'Too slow'
%   ITI           1.0-1.5 s (jittered, from the taskset), dimmed fixation
%
% Every design choice (fractals, rule sequence, side of A, frequencies, key
% mapping, trial order, ITIs) comes from the subject's taskset, made offline:
%   makeTaskset(subjID)
%
% Inputs:
%   subjID     numeric subject id (zero-padded to 2 digits)
%   startBlock first block to run, default 1 (e.g. 2 to resume after a crash);
%              the session then runs through block parameters.nBlocks
%   debugMode  1 = transparent window + SkipSyncTests, keyboard not captured,
%              refresh-rate mismatch only warns
%              0 = full-screen, timing enforced (default)
%
% EEG trigger codes (see sendTrigger.m):
%   0 block start | 1 fixation | 21-24 stimulus onset, compound 1-4
%   3 stimulus offset = response window | 41/42 response F/J
%   51/52/53 feedback correct/incorrect/timeout | 6 ITI | 7 block end
%
% Press ESCAPE during the stimulus, a response window or an end-of-block
% screen to abort; data collected so far is still saved.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 1 || isempty(subjID); error('catlearn_task: subjID is required.'); end
if nargin < 2 || isempty(startBlock); startBlock = 1; end
if nargin < 3 || isempty(debugMode); debugMode = 0; end

close all; clc;
if ischar(subjID) || isstring(subjID); subjID = str2double(subjID); end
subjID = num2str(subjID, '%02d'); % convert subjID to string

% Resolve paths relative to this file, so the task runs from any cwd
taskDir = fileparts(mfilename('fullpath'));
addpath(taskDir);

% Add Psychtoolbox if it is not already on the path. Extend the candidate
% list (or add a hostname branch) for the recording rig.
if exist('Screen', 'file') ~= 3 && exist('Screen', 'file') ~= 2
    % The first entry is the inner toolbox folder of a git clone of
    % Psychtoolbox-3; the repo root also holds C/Python sources that must
    % not go on the MATLAB path.
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
    error('catlearn_task: Psychtoolbox not found on the path.')
end

% Load settings and the subject's offline taskset
parameters = loadParameters(subjID);
parameters.isDemoMode = logical(debugMode);
parameters.debugMode = logical(debugMode);
if startBlock < 1 || startBlock > parameters.nBlocks
    error('catlearn_task: startBlock must be 1-%d.', parameters.nBlocks)
end

tasksetFile = fullfile(parameters.tasksetDir, ['taskset_sub' subjID '.mat']);
if exist(tasksetFile, 'file') ~= 2
    error('catlearn_task: no taskset for subject %s (%s). Run makeTaskset(%d) first.', ...
        subjID, tasksetFile, str2double(subjID))
end
taskset = load(tasksetFile);
taskset = taskset.taskset;
if size(taskset.fractals, 1) ~= parameters.nBlocks || ~isfield(taskset.blocks, 'fractals')
    error(['catlearn_task: %s is in an old format (not 4 new fractals per block). ' ...
        'Regenerate it with makeTaskset(%d, true).'], tasksetFile, str2double(subjID))
end
fprintf('sub%s: rules %s, A on the %s, %d Hz left / %d Hz right, category 0 = %s, 1 = %s\n', ...
    subjID, strjoin(taskset.rules, '-'), taskset.sideA, taskset.freqLeftRight, ...
    upper(taskset.categoryKeys{1}), upper(taskset.categoryKeys{2}));

% Fail loudly here rather than mid-block if EEG is on but unconfigured
if parameters.EEG && strcmp(parameters.triggerMethod, 'none')
    error('catlearn_task: parameters.EEG = 1 but triggerMethod is ''none''. Configure sendTrigger.m first.')
end

data_path = fullfile(parameters.dataDir, ['sub' subjID]);

if debugMode
    Screen('Preference', 'SkipSyncTests', 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Open the screen and peripherals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Everything past this point runs inside try/catch so a crash cannot leave a
% black full-screen window and a captured keyboard behind.
kbx = [];
stim = [];
try
    screen = initScreen(parameters);
    [kbx, parameters] = initPeripherals(parameters);
    stim = initStimuli(parameters, screen, taskset);
    fprintf('refresh %.2f Hz: %s Hz tagging = %s frames per cycle, %d frames per stimulus\n', ...
        stim.refreshHz, mat2str(stim.freqActual, 4), mat2str(stim.framesPerCycle), stim.nStimFrames);

    if ~debugMode
        HideCursor(screen.id);
        ListenChar(-1); % stop keypresses leaking into the MATLAB console
    end

    KbQueueStart(kbx);
    KbQueueFlush(kbx);
    showprompts(screen, 'WelcomeWindow');
    waitForKey(kbx, parameters.space_key);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Blocks
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for block = startBlock:parameters.nBlocks
        parameters = initFiles(parameters, screen, data_path, kbx, block);
        KbQueueStart(kbx); % initFiles stops the queue after its overwrite prompt
        tMap = makeTaskMap(parameters, taskset, block);

        [timeReport, respReport, EEGtimeReport, trigReport, aborted, lastTrial] = ...
            runBlock(parameters, screen, stim, kbx, tMap, block);

        % Save the block
        if lastTrial > 0
            accuracy = 100 * mean([respReport(1:lastTrial).correct]);
            medRT = median([respReport(1:lastTrial).RT], 'omitnan');
            nTimeout = sum([respReport(1:lastTrial).timedOut]);
            nMissed = sum([timeReport(1:lastTrial).missedFrames]);
        else
            respReport = struct([]); % no completed trials, keep the file honest
            accuracy = NaN; medRT = NaN; nTimeout = 0; nMissed = 0;
        end
        fprintf(['\nBlock %02d (rule %s) done: %d/%d trials, accuracy %.1f%%, median RT %.3f s after offset, ' ...
                 '%d timeouts, %d missed frames\n'], block, tMap.rule, lastTrial, tMap.nTrials, accuracy, ...
                medRT, nTimeout, nMissed);

        matFile.parameters = parameters;
        matFile.screen = screen;
        matFile.stim = rmfield(stim, 'fracTex'); % texture handles are meaningless once closed
        matFile.taskset = taskset;
        matFile.tMap = tMap;
        matFile.timeReport = timeReport;
        matFile.respReport = respReport;
        matFile.aborted = aborted;
        matFile.nTrialsCompleted = lastTrial;
        save([parameters.block_dir filesep parameters.matFile], 'matFile')

        EEGsummary.timeReport = EEGtimeReport;
        EEGsummary.trigReport = trigReport;
        save([parameters.block_dir filesep parameters.EEGreportFile], 'EEGsummary')
        fprintf('Saved to %s\n', parameters.block_dir);

        if aborted
            break;
        end

        % End-of-block screen
        showprompts(screen, 'BlockEnd', accuracy);
        % Cosmetic only, and rigs often have no audio device configured --
        % never let it take down a block whose data is already saved.
        try Beeper('med', 0.5, 0.1); catch; end %#ok<CTCH>
        WaitSecs(2);
        if block < parameters.nBlocks
            key = waitForKey(kbx, {parameters.space_key, parameters.exit_key}, ...
                screen, 'ContinueorEsc', block);
            if strcmp(key, parameters.exit_key)
                fprintf('Session ended by experimenter after block %02d.\n', block);
                break;
            end
        end
    end

    showprompts(screen, 'EndExperiment');
    WaitSecs(2);
    cleanUp(kbx, stim, debugMode);

catch ME
    cleanUp(kbx, stim, debugMode);
    rethrow(ME);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% One block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [timeReport, respReport, EEGtimeReport, trigReport, aborted, lastTrial] = ...
    runBlock(parameters, screen, stim, kbx, tMap, block)
nTrials = tMap.nTrials;
timeReport = struct;  % flip timestamps, one entry per trial
respReport = struct;  % subject responses
EEGtimeReport = struct;
% Preallocated so appending a trigger code never allocates on the trial
% timeline: up to 6 per trial (fix, stim, offset, respKey, feedback, iti)
% plus block start/end. Trailing NaNs are trimmed before returning.
trigReport = NaN(1, nTrials*6 + 2);
trig_counter = 1;

exitCode = KbName(parameters.exit_key);
respCodes = KbName(parameters.resp_keys);

EEGtimeReport.blockstart = sendTrigger(parameters, 0);
trigReport(trig_counter) = 0; trig_counter = trig_counter + 1;

showprompts(screen, 'BlockStart', block);
WaitSecs(2);

aborted = false;
lastTrial = 0;
nextFixOnset = 0; % 0 == flip at the next retrace
nF = stim.nStimFrames;

for trial = 1:nTrials
    s = tMap.stim(trial);
    fprintf('block %02d (%s), trial %02d, compound %d ... ', block, tMap.rule, trial, s);

    % Textures for this compound, placed by the subject's side of A
    texLR = zeros(1, 2);
    texLR(stim.rectA) = stim.fracTex(block, 1 + tMap.levelA(trial)); % this block's a1 or a2
    texLR(stim.rectB) = stim.fracTex(block, 3 + tMap.levelB(trial)); % this block's b1 or b2

    %------------------------------------------------------------------
    % Fixation window (1 s)
    %------------------------------------------------------------------
    drawTextures(parameters, screen, stim, 'FixationCross');
    tFixOn = Screen('Flip', screen.win, nextFixOnset);
    EEGtimeReport.fix(trial) = sendTrigger(parameters, 1);
    trigReport(trig_counter) = 1; trig_counter = trig_counter + 1;
    timeReport(trial).fixOnset = tFixOn;

    %------------------------------------------------------------------
    % Stimulus window (2 s of flicker, one flip per frame, fixation on)
    %------------------------------------------------------------------
    % Contrast is indexed by FRAME, so the flicker stays locked to the
    % display even if a frame is late; late frames are counted below.
    flipTimes = NaN(1, nF);
    when = tFixOn + parameters.fixDuration - screen.ifi/2;
    for k = 1:nF
        drawTextures(parameters, screen, stim, 'Fractals', [], texLR, stim.contrast(:, k)');
        drawTextures(parameters, screen, stim, 'FixationCross');
        flipTimes(k) = Screen('Flip', screen.win, when);
        if k == 1
            EEGtimeReport.stim(trial) = sendTrigger(parameters, 20 + s);
            trigReport(trig_counter) = 20 + s; trig_counter = trig_counter + 1;
            KbQueueFlush(kbx); % keys from here to offset are early presses
        end
        when = flipTimes(k) + screen.ifi/2; % next retrace
    end

    drawTextures(parameters, screen, stim, 'FixationCross');
    tStimOff = Screen('Flip', screen.win, when);
    EEGtimeReport.resp(trial) = sendTrigger(parameters, 3);
    trigReport(trig_counter) = 3; trig_counter = trig_counter + 1;
    tStimOn = flipTimes(1);
    timeReport(trial).stimOnset = tStimOn;
    timeReport(trial).stimOffset = tStimOff;
    timeReport(trial).fixDuration = tStimOn - tFixOn;
    timeReport(trial).stimDuration = tStimOff - tStimOn;
    timeReport(trial).flipTimes = flipTimes;
    timeReport(trial).missedFrames = sum(diff([flipTimes tStimOff]) > 1.5 * screen.ifi);

    % Keys during the flicker: ESCAPE aborts, F/J are logged but do not count
    [pressed, firstPress] = KbQueueCheck(kbx);
    if pressed && firstPress(exitCode) > 0
        aborted = true;
        fprintf('aborted by experimenter.\n');
        break;
    end
    earlyPress = pressed && any(firstPress(respCodes) > 0);
    KbQueueFlush(kbx);

    %------------------------------------------------------------------
    % Response window (deadline measured from stimulus offset)
    %------------------------------------------------------------------
    respKey = '';
    respTime = NaN;
    respDeadline = tStimOff + parameters.respDuration;
    while GetSecs < respDeadline
        % firstPress holds a GetSecs timestamp per keycode, so indexing it
        % by keycode gives exact press times. Check the response keys as a
        % group rather than taking the earliest of ALL pressed keys -- one
        % KbQueueCheck can report several keys at once, and picking a
        % non-response key would discard the real response with it.
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
                respTime = pressTime - tStimOff;
                break;
            end
        end
    end
    if aborted
        fprintf('aborted by experimenter.\n');
        break;
    end

    if isempty(respKey)
        respCode = 0; % no key logged, so no response trigger
    elseif strcmp(respKey, 'f')
        respCode = 41;
    else
        respCode = 42;
    end
    if respCode > 0
        EEGtimeReport.respKey(trial) = sendTrigger(parameters, respCode);
        trigReport(trig_counter) = respCode; trig_counter = trig_counter + 1;
    else
        EEGtimeReport.respKey(trial) = NaN;
    end

    timedOut = isempty(respKey);
    correct = ~timedOut && strcmpi(respKey, tMap.correctKey{trial});

    respReport(trial).key = respKey;
    respReport(trial).correctKey = tMap.correctKey{trial};
    respReport(trial).RT = respTime;                          % s from stimulus OFFSET
    respReport(trial).RTfromOnset = respTime + (tStimOff - tStimOn);
    respReport(trial).correct = correct;
    respReport(trial).timedOut = timedOut;
    respReport(trial).earlyPress = earlyPress;                % F/J during the flicker (ignored)
    respReport(trial).rule = tMap.rule;
    respReport(trial).stim = s;
    respReport(trial).levelA = tMap.levelA(trial);
    respReport(trial).levelB = tMap.levelB(trial);
    respReport(trial).category = tMap.category(trial);
    timeReport(trial).respDuration = GetSecs - tStimOff;
    lastTrial = trial; % only counted once the trial's data is on record

    %------------------------------------------------------------------
    % Feedback window (0.5 s), shown as soon as the response comes in
    %------------------------------------------------------------------
    if timedOut
        showprompts(screen, 'Timeout', [], 0, parameters.timeoutColor);
        fbCode = 53;
    elseif correct
        showprompts(screen, 'Correct', [], 0, parameters.correctColor);
        fbCode = 51;
    else
        showprompts(screen, 'Incorrect', [], 0, parameters.incorrectColor);
        fbCode = 52;
    end
    tFbOn = Screen('Flip', screen.win);
    EEGtimeReport.feedback(trial) = sendTrigger(parameters, fbCode);
    trigReport(trig_counter) = fbCode; trig_counter = trig_counter + 1;
    timeReport(trial).feedbackOnset = tFbOn;

    %------------------------------------------------------------------
    % Intertrial window (jittered)
    %------------------------------------------------------------------
    drawTextures(parameters, screen, stim, 'FixationCrossITI');
    tItiOn = Screen('Flip', screen.win, tFbOn + parameters.feedbackDuration - screen.ifi/2);
    EEGtimeReport.iti(trial) = sendTrigger(parameters, 6);
    trigReport(trig_counter) = 6; trig_counter = trig_counter + 1;
    timeReport(trial).itiOnset = tItiOn;
    timeReport(trial).feedbackDuration = tItiOn - tFbOn;

    % The ITI ends when the next trial's fixation flips
    nextFixOnset = tItiOn + tMap.iti(trial) - screen.ifi/2;
    timeReport(trial).itiRequested = tMap.iti(trial);

    if timedOut
        fprintf('no response');
    else
        verdict = {'incorrect', 'correct'};
        fprintf('%s, %s, RT = %.3f s', upper(respKey), verdict{correct+1}, respTime);
    end
    if earlyPress; fprintf(' (early press)'); end
    if timeReport(trial).missedFrames > 0
        fprintf(' [%d missed frames]', timeReport(trial).missedFrames);
    end
    fprintf('\n');
end

% Hold the last ITI so trial N gets the same post-feedback interval as
% every other trial before the end-of-block screen appears
if ~aborted && lastTrial > 0
    WaitSecs('UntilTime', nextFixOnset);
end

EEGtimeReport.blockend = sendTrigger(parameters, 7);
trigReport(trig_counter) = 7; trig_counter = trig_counter + 1;
trigReport = trigReport(1:trig_counter-1); % drop unused preallocated slots
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local helpers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pressedKey = waitForKey(kbx, acceptKeys, screen, promptName, promptNum)
% Blocks until one of acceptKeys is pressed. If a prompt is given it is
% redrawn while waiting (the srh_ori pattern), otherwise whatever is already
% on screen stays up.
if ischar(acceptKeys); acceptKeys = {acceptKeys}; end
redraw = nargin >= 4 && ~isempty(promptName);
if redraw && nargin < 5; promptNum = []; end

pressedKey = '';
acceptCodes = KbName(acceptKeys);
KbQueueFlush(kbx);
while isempty(pressedKey)
    if redraw
        showprompts(screen, promptName, promptNum);
    end
    [pressed, firstPress] = KbQueueCheck(kbx);
    if pressed
        pressTimes = firstPress(acceptCodes);
        hit = find(pressTimes > 0);
        if ~isempty(hit)
            [~, iEarliest] = min(pressTimes(hit));
            pressedKey = acceptKeys{hit(iEarliest)};
        end
    end
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
if ~isempty(stim) && isstruct(stim) && isfield(stim, 'fracTex')
    try Screen('Close', stim.fracTex(:)'); catch; end %#ok<CTCH>
end
sca;
Priority(0);
if debugMode
    PsychDebugWindowConfiguration(0, 1); % undo the transparent-window setting
end
end
