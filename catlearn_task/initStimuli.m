function stim = initStimuli(parameters, screen, taskset)
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
%
% Builds the subject's fractal textures (4 per block, new ones every block:
% stim.fracTex(block, [a1 a2 b1 b2])), the destination rects and the
% frame-by-frame flicker ONCE per session, so nothing is created on the
% trial timeline and no texture handles leak.
%
% Size: every fractal PNG is cropped to the same square -- the smallest one
% enclosing the visible (alpha > 0) part of ALL fractals in the folder -- and
% that square is drawn parameters.fractalSizeDeg wide. Fractals therefore keep
% their relative sizes and none is larger than fractalSizeDeg.
%
% Flicker: sinusoidal contrast, c(k) = (1 - cos(2*pi*(k-1)/framesPerCycle))/2
% on stimulus frame k, so each fractal starts invisible, peaks at full
% contrast mid-cycle, and completes a whole number of cycles in stimDuration.
% Contrast is applied as the texture's global alpha: over the grey background
% this scales the fractal's deviation from grey, keeping its own alpha mask.
%
% Call Screen('Close', stim.fracTex) -- or just sca -- to release the textures.

% Pixels per degree at the current viewing distance. va2pixel returns a
% half-extent, so the full extent of a 1 deg object is 2*va2pixel(1).
stim.pixPerDeg = 2 * va2pixel(parameters, screen, 1);
stim.sizePix = 2 * va2pixel(parameters, screen, parameters.fractalSizeDeg); % drawn side, pixels
stim.eccPix = va2pixel(parameters, screen, 2 * parameters.eccentricityDeg); % horizontal offset, pixels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fractal textures, one row per block: [a1 a2 b1 b2]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stim.crop = sharedCrop(parameters);                 % [row0 row1 col0 col1], pixels of the PNG
stim.fractals = taskset.fractals;
stim.fracTex = zeros(size(taskset.fractals));
for b = 1:size(taskset.fractals, 1)
    for ii = 1:4
        [img, ~, alpha] = imread(fullfile(parameters.fractalDir, sprintf('%d.png', taskset.fractals(b, ii))));
        if isempty(alpha); alpha = 255 * ones(size(img, 1), size(img, 2), 'uint8'); end
        rgba = cat(3, img, alpha);
        rgba = rgba(stim.crop(1):stim.crop(2), stim.crop(3):stim.crop(4), :);
        stim.fracTex(b, ii) = Screen('MakeTexture', screen.win, rgba);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Destination rects: column 1 = left of fixation, column 2 = right
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
baseRect = [0 0 stim.sizePix stim.sizePix];
stim.rects = [ ...
    CenterRectOnPoint(baseRect, screen.xCenter - stim.eccPix, screen.yCenter)', ...
    CenterRectOnPoint(baseRect, screen.xCenter + stim.eccPix, screen.yCenter)'];
% Which texture slot (1-4 = a1 a2 b1 b2) goes on the left/right is decided per
% trial; which SIDE shows dimension A is fixed per subject.
stim.sideA = taskset.sideA;
if strcmp(taskset.sideA, 'left')
    stim.rectA = 1; stim.rectB = 2;
else
    stim.rectA = 2; stim.rectB = 1;
end

xEdges = stim.rects([1 3], :);
yEdges = stim.rects([2 4], :);
if any(stim.rects(:) < 0) || any(xEdges(:) > screen.screenXpixels) ...
        || any(yEdges(:) > screen.screenYpixels)
    warning('initStimuli: a fractal falls (partly) off screen. Check eccentricityDeg/fractalSizeDeg/viewingDistance.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Flicker: whole frames per cycle at this refresh rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stim.refreshHz = 1 / screen.ifi;
stim.freqLeftRight = taskset.freqLeftRight;          % Hz, [left right]
fpc = stim.refreshHz ./ stim.freqLeftRight;          % frames per cycle
stim.framesPerCycle = round(fpc);
stim.nStimFrames = round(parameters.stimDuration * stim.refreshHz);
bad = abs(fpc - stim.framesPerCycle) > parameters.refreshTolerance | ...
      mod(stim.nStimFrames, stim.framesPerCycle) ~= 0;
if any(bad)
    msg = sprintf(['initStimuli: at %.2f Hz refresh, %s Hz is %s frames per cycle and %d stimulus frames ' ...
        'are not whole cycles. Use tagging frequencies that divide the refresh rate.'], ...
        stim.refreshHz, mat2str(stim.freqLeftRight), mat2str(fpc, 3), stim.nStimFrames);
    if parameters.debugMode
        warning('%s Continuing in debug mode with rounded frames per cycle.', msg);
    else
        error('%s', msg);
    end
end
stim.freqActual = stim.refreshHz ./ stim.framesPerCycle; % what is really shown, Hz

% Contrast of the left/right fractal on every stimulus frame: (2, nStimFrames)
k = 0:stim.nStimFrames - 1;
stim.contrast = [(1 - cos(2 * pi * k / stim.framesPerCycle(1))) / 2; ...
                 (1 - cos(2 * pi * k / stim.framesPerCycle(2))) / 2];
end

function crop = sharedCrop(parameters)
% Smallest centered-ish square containing the visible part of every fractal
rows = [inf -inf]; cols = [inf -inf];
for k = 1:parameters.nFractals
    [~, ~, alpha] = imread(fullfile(parameters.fractalDir, sprintf('%d.png', k)));
    if isempty(alpha); continue; end
    r = find(any(alpha > 0, 2)); c = find(any(alpha > 0, 1));
    rows = [min(rows(1), r(1)) max(rows(2), r(end))];
    cols = [min(cols(1), c(1)) max(cols(2), c(end))];
end
if ~all(isfinite([rows cols]))
    error('initStimuli: no fractal with an alpha channel found in %s', parameters.fractalDir)
end
side = max(diff(rows), diff(cols)) + 1;
r0 = round(mean(rows) - (side - 1) / 2); c0 = round(mean(cols) - (side - 1) / 2);
r0 = max(1, r0); c0 = max(1, c0); % every PNG is the same size, so the crop fits them all
crop = [r0, r0 + side - 1, c0, c0 + side - 1];
end
