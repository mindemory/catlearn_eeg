function parameters = loadParameters(subjID)
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
% All experiment-wide settings live here. Nothing in this file touches
% hardware, so it is safe to call before the screen is opened.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% program basic settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.transparency = 0.85;  % transparency of the PTB window in debug mode
parameters.viewingDistance = 55; % viewDist (in cm)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% study parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.studyname = 'catlearn_eeg';
parameters.subject = subjID;
parameters.nTrials = 40;         % trials per block (multiple of 4: each compound equally often)
parameters.nBlocks = 3;          % blocks per session, one rule each

% Data lives outside the code repo, under ~/Documents/data/catlearn_eeg on
% every machine. Built from $HOME so the same code works on the Mac and the
% Linux rig; override here if the rig keeps data elsewhere.
parameters.dataDir = fullfile(getenv('HOME'), 'Documents', 'data', 'catlearn_eeg', 'catlearn_task');
% Per-subject tasksets written offline by makeTaskset.m
parameters.tasksetDir = fullfile(parameters.dataDir, 'tasksets');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Design (2x2): location A shows fractal a1 or a2, location B shows b1 or b2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rule sequences across the 3 blocks. Category of a compound under each rule:
%   A  -> which A fractal is shown, B -> which B fractal, AB -> A xor B.
% makeTaskset counterbalances sequence, side of A, tagging frequencies and
% key mapping across subject IDs.
parameters.ruleSequences = {{'A', 'A', 'AB'}, {'A', 'B', 'AB'}, {'AB', 'AB', 'A'}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stimulus parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Two fractals flanking fixation on the horizontal meridian, each flickering
% (sinusoidal contrast) at its own tagging frequency.
parameters.fractalDir = fullfile(fileparts(mfilename('fullpath')), 'fractals'); % 1.png ... 72.png, RGBA
parameters.nFractals = 72;
parameters.fractalSizeDeg = 4;   % side of the square that encloses every fractal, dva (= old grating diameter)
parameters.eccentricityDeg = 6;  % horizontal offset of each fractal center from fixation, dva
parameters.tagFreqs = [12 15];   % Hz; must divide the refresh rate (60 Hz: 5 and 4 frames per cycle)
parameters.refreshTolerance = 0.05; % max deviation of refresh/freq from a whole number of frames

% Fixation
parameters.fixationSizeDeg = 0.6; % outer extent of the fixation cross in dva

% Colors (0-255; screen.white is 255 with the default PTB color range)
parameters.fixColor = [255 255 255];      % white
parameters.correctColor = [0 255 0];      % green
parameters.incorrectColor = [255 0 0];    % red
parameters.timeoutColor = [255 191 0];    % amber

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% timing parameters (in seconds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.fixDuration = 1.0;      % fixation alone before the fractals appear
parameters.stimDuration = 2.0;     % flickering fractals, fixation stays up; whole cycles of both frequencies
parameters.respDuration = 2.0;     % response window, measured from STIMULUS OFFSET (responses only after the flicker)
parameters.feedbackDuration = 0.5; % 'Correct' / 'Incorrect' / 'Too slow' on screen
parameters.itiDuration = [1.0 1.5];% jitter range, uniformly sampled per trial

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EEG / eyetracking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set parameters.EEG = 1 once a trigger method is implemented in sendTrigger.m.
% While EEG = 0 the trigger calls are no-ops that still record timestamps, so
% turning the flag on later requires no changes to catlearn_task.m.
parameters.EEG = 0;
parameters.triggerMethod = 'none'; % 'none' | 'parallel' | 'serial' | 'python'
parameters.triggerPort = '';       % e.g. 'D010' (parallel) or '/dev/ttyUSB0' (serial)
parameters.eyetracker = 0;         % not wired up in this task yet
end
