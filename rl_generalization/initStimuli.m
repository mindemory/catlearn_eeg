function stim = initStimuli(parameters, screen, subjectInfo, config)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
%
% Loads the subject's four noise patches as textures and works out where
% the two stimulus locations fall on screen for the chosen spatial
% configuration. Done once per session, off the trial timeline.
%
% stim.tex = [a1 a2 b1 b2]: dimension-A level k is stim.tex(k), dimension-B
% level k is stim.tex(2 + k). stim.rects(:, loc) is location A (loc = 1) or
% B (loc = 2). Release with Screen('Close', stim.tex) or sca.

[stim.cells, stim.configName] = spatialConfig(config);
stim.config = config;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Textures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The PNGs are grayscale stored as RGB, fully opaque, on a 128 background
% (matching the window's mid-grey), so only one channel is needed.
stim.tex = zeros(1, 4);
for k = 1:4
    img = imread(subjectInfo.stimFiles{k});
    if size(img, 3) > 1
        img = img(:, :, 1);
    end
    stim.tex(k) = Screen('MakeTexture', screen.win, img);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Locations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The screen's shorter side is split into gridSize square cells, so the
% grid spans the full screen height with fixation in the center cell.
% Positions are whole multiples of a cell from fixation. Grid rows grow
% downward, the same direction as screen y.
stim.cellPix = floor(min(screen.screenXpixels, screen.screenYpixels) / parameters.gridSize);
stim.sizePix = round(parameters.stimFraction * stim.cellPix);
fixCell = (parameters.gridSize - 1) / 2;
baseRect = [0 0 stim.sizePix stim.sizePix];
stim.centers = zeros(2, 2); % [x; y] per location
stim.rects = zeros(4, 2);
for loc = 1:2
    offsetCells = stim.cells(loc, :) - fixCell; % [down right]
    stim.centers(:, loc) = [screen.xCenter + offsetCells(2) * stim.cellPix; ...
                            screen.yCenter + offsetCells(1) * stim.cellPix];
    stim.rects(:, loc) = CenterRectOnPoint(baseRect, stim.centers(1, loc), stim.centers(2, loc))';
end

% What that works out to in visual angle on this monitor at this viewing
% distance -- no longer set directly, so record it with the data
pix2deg = @(px) atand(px * screen.pixSize / parameters.viewingDistance);
stim.cellSizeDeg = pix2deg(stim.cellPix);  % width of one cell, measured from fixation
stim.sizeDeg = 2 * pix2deg(stim.sizePix / 2); % patch width, centered on fixation
stim.eccDeg = arrayfun(@(loc) pix2deg(norm(stim.centers(:, loc) - [screen.xCenter; screen.yCenter])), 1:2);
fprintf('initStimuli: grid cell %d px (%.2f deg), patch %d px (%.2f deg), eccentricity A %.2f / B %.2f deg\n', ...
    stim.cellPix, stim.cellSizeDeg, stim.sizePix, stim.sizeDeg, stim.eccDeg);

xEdges = stim.rects([1 3], :);
yEdges = stim.rects([2 4], :);
if any(stim.rects(:) < 0) || any(xEdges(:) > screen.screenXpixels) ...
        || any(yEdges(:) > screen.screenYpixels)
    % The grid is sized to fit the screen, so this only fires if a patch is
    % set larger than its cell
    warning('initStimuli: config %d (%s) puts a patch (partly) off screen. Check parameters.stimFraction (<= 1).', ...
        config, stim.configName)
end
end
