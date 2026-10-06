% testStimuli.m
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
%
% Standalone check of the display, with no trial logic. Opens a transparent
% debug window, prints the derived geometry and tagging frames, then shows
% one compound of block 1 of subject 99's taskset: first static at full contrast, then
% flickering for parameters.stimDuration, reporting missed frames.
%
% Press any key to start the flicker, and again to quit.
clear; close all; clc; sca;

taskDir = fileparts(mfilename('fullpath'));
addpath(taskDir);
if exist('Screen', 'file') == 0
    ptbCandidates = {fullfile(getenv('HOME'), 'Documents', 'MATLAB', 'Psychtoolbox', 'Psychtoolbox'), ...
                     '/Applications/Psychtoolbox', ...
                     '/usr/share/psychtoolbox-3'};
    for ii = 1:numel(ptbCandidates)
        if exist(ptbCandidates{ii}, 'dir') == 7
            addpath(genpath(ptbCandidates{ii})); break;
        end
    end
end
Screen('Preference', 'SkipSyncTests', 1);

parameters = loadParameters('99');
parameters.isDemoMode = true;
parameters.debugMode = true;
parameters.transparency = 1;
taskset = makeTaskset(99); % deterministic, saved as taskset_sub99.mat if new

screen = initScreen(parameters);
stim = initStimuli(parameters, screen, taskset);

fprintf('screen        : %d x %d px, %.1f x %.1f cm\n', ...
    screen.screenXpixels, screen.screenYpixels, screen.screenWidth, screen.screenHeight);
fprintf('screen extent : %.1f x %.1f dva at %d cm\n', ...
    screen.deg_width, screen.deg_height, parameters.viewingDistance);
fprintf('pixels/degree : %.2f\n', stim.pixPerDeg);
fprintf('fractal       : %.1f dva = %d px square (PNG crop rows %d-%d, cols %d-%d)\n', ...
    parameters.fractalSizeDeg, stim.sizePix, stim.crop);
fprintf('eccentricity  : %.1f dva = %d px offset\n', parameters.eccentricityDeg, stim.eccPix);
fprintf('fractals      : block 1 a1 %d, a2 %d, b1 %d, b2 %d; A on the %s\n', taskset.fractals(1, :), taskset.sideA);
fprintf('refresh       : %.2f Hz (ifi %.4f s)\n', stim.refreshHz, screen.ifi);
fprintf('tagging       : left %g Hz, right %g Hz -> %s frames per cycle, shown at %s Hz\n', ...
    stim.freqLeftRight, mat2str(stim.framesPerCycle), mat2str(stim.freqActual, 4));
fprintf('stimulus      : %d frames = %.3f s\n', stim.nStimFrames, stim.nStimFrames * screen.ifi);

texLR = zeros(1, 2);
texLR(stim.rectA) = stim.fracTex(1, 1); % block 1 a1
texLR(stim.rectB) = stim.fracTex(1, 3); % block 1 b1

drawTextures(parameters, screen, stim, 'Fractals', [], texLR, [1 1]);
drawTextures(parameters, screen, stim, 'FixationCross');
Screen('Flip', screen.win);
KbStrokeWait;

flipTimes = NaN(1, stim.nStimFrames);
when = 0;
for k = 1:stim.nStimFrames
    drawTextures(parameters, screen, stim, 'Fractals', [], texLR, stim.contrast(:, k)');
    drawTextures(parameters, screen, stim, 'FixationCross');
    flipTimes(k) = Screen('Flip', screen.win, when);
    when = flipTimes(k) + screen.ifi/2;
end
drawTextures(parameters, screen, stim, 'FixationCross');
tOff = Screen('Flip', screen.win, when);
fprintf('flicker       : %.3f s, %d missed frames\n', tOff - flipTimes(1), ...
    sum(diff([flipTimes tOff]) > 1.5 * screen.ifi));

KbStrokeWait;
Screen('Close', stim.fracTex(:)');
sca;
Priority(0);
PsychDebugWindowConfiguration(0, 1);
