function parameters = loadParameters(subjID)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
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
parameters.studyname = 'rl_generalization';
parameters.subject = subjID;
parameters.nTrials = 40; % trials per block; must be a multiple of 4 (each compound equally often)

% Data lives outside the code repo, under ~/Documents/data/catlearn_eeg on
% every machine. Built from $HOME so the same code works on the Mac and the
% Linux rig; override here if the rig keeps data elsewhere.
dataRoot = fullfile(getenv('HOME'), 'Documents', 'data', 'catlearn_eeg');
parameters.dataDir = fullfile(dataRoot, 'rl_generalization');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stimulus parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Four noise patches per subject -- two levels for location A, two for
% location B -- drawn once from this pool and reused for every block of
% that subject (see initSubject.m). The rig needs a copy of the PNGs at the
% same place under its own $HOME.
parameters.stimDir = fullfile(dataRoot, 'task_design', 'stimuli', 'png');
parameters.stimPattern = 'noisepatch_%02d.png';
parameters.nStimPool = 20;

% Spatial layout: the screen's shorter side is divided into a 7x7 grid of
% square cells with fixation in the center cell, so the grid spans the full
% screen height (the extra width of a landscape screen stays empty). The
% spatial configuration picks which two cells hold the stimuli
% (spatialConfig.m). Cell and patch sizes in dva therefore depend on the
% monitor and viewing distance; initStimuli reports and saves them.
parameters.gridSize = 7;        % cells per side
parameters.stimFraction = 0.9;  % patch width as a fraction of a cell (0.05-cell inset per side, as in the design figure)

% Fixation
parameters.fixationSizeDeg = 0.6; % outer extent of the fixation cross in dva

% Colors (0-255; screen.white is 255 with the default PTB color range)
parameters.fixColor = [255 255 255];    % white cross during fixation and stimulus
parameters.correctColor = [0 255 0];    % green
parameters.incorrectColor = [255 0 0];  % red
parameters.timeoutColor = [255 191 0];  % amber

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% timing parameters (in seconds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.fixDuration = 1.0;      % white fixation alone
parameters.stimDuration = 4.0;     % stimuli stay up until a response or this deadline
parameters.feedbackDuration = 1.5; % 'Correct' / 'Incorrect' / 'Too slow', then straight to the next fixation
end
