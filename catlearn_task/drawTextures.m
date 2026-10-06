function drawTextures(parameters, screen, stim, texture_name, color, texLR, alphaLR)
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
%
% Draws one component into the backbuffer. Unlike the srh_ori version this
% function never calls Screen('Flip') -- the caller owns the flip, which is
% what lets catlearn_task.m schedule flips against a deadline and get exact
% 1 s / 200 ms durations.
%
% texture_name:
%   'FixationCross'     bullseye+cross fixation (Thaler et al., 2013 style)
%   'FixationCrossITI'  same, dimmed, so the ITI is visually distinct
%   'Fractals'          the two fractals: texLR = texture handles [left right],
%                       alphaLR = their contrast (0-1) on this frame
if nargin < 5 || isempty(color)
    color = parameters.fixColor;
end

switch texture_name

    % Drawing Fixation Cross
    case {'FixationCross', 'FixationCrossITI'}
        if strcmp(texture_name, 'FixationCrossITI')
            discColor = (screen.black + screen.grey)/2; % dimmed disc during the ITI
        else
            discColor = screen.black;
        end

        % Pixel radii for the outer disc, the cross arms and the inner dot
        r_pix_outer = va2pixel(parameters, screen, parameters.fixationSizeDeg);
        r_pix_inner = va2pixel(parameters, screen, parameters.fixationSizeDeg/3);

        baseRect_outer = [0 0 r_pix_outer*2 r_pix_outer*2];
        centeredRect_outer = CenterRectOnPoint(baseRect_outer, screen.xCenter, screen.yCenter);
        baseRect_inner = [0 0 r_pix_inner*2 r_pix_inner*2];
        centeredRect_inner = CenterRectOnPoint(baseRect_inner, screen.xCenter, screen.yCenter);

        % Cross arms spanning the outer disc
        xCoords = [-r_pix_outer r_pix_outer 0 0];
        yCoords = [0 0 -r_pix_outer r_pix_outer];
        allCoords = [xCoords; yCoords];

        Screen('FillOval', screen.win, discColor, centeredRect_outer);
        Screen('DrawLines', screen.win, allCoords, max(1, round(r_pix_inner*1.5)), ...
            color, [screen.xCenter screen.yCenter], 2); % 2 == high quality smoothing
        Screen('FillOval', screen.win, discColor, centeredRect_inner);

    % Drawing the two fractals
    case 'Fractals'
        % Global alpha = contrast: over the grey background, alpha blending
        % scales each fractal's deviation from grey (see initStimuli).
        Screen('DrawTextures', screen.win, texLR, [], stim.rects, [], [], alphaLR);

    otherwise
        error('drawTextures: unknown texture_name "%s".', texture_name)
end
end
