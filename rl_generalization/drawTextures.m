function drawTextures(parameters, screen, stim, texture_name, opt)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
%
% Draws one component into the backbuffer. Never calls Screen('Flip') --
% the caller owns the flip, so flips can be scheduled against deadlines.
%
% texture_name:
%   'FixationCross'     bullseye+cross fixation (Thaler et al., 2013 style);
%                       opt = cross color, default parameters.fixColor
%   'Stimuli'           both patches; opt = [levelA levelB], the level (1 or
%                       2) of dimension A shown at location A and of
%                       dimension B shown at location B
if nargin < 5
    opt = [];
end

switch texture_name

    % Drawing Fixation Cross
    case 'FixationCross'
        if isempty(opt)
            color = parameters.fixColor;
        else
            color = opt;
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

        Screen('FillOval', screen.win, screen.black, centeredRect_outer);
        Screen('DrawLines', screen.win, allCoords, max(1, round(r_pix_inner*1.5)), ...
            color, [screen.xCenter screen.yCenter], 2); % 2 == high quality smoothing
        Screen('FillOval', screen.win, screen.black, centeredRect_inner);

    % Drawing the two stimuli
    case 'Stimuli'
        % One DrawTextures call: the dimension-A texture into location A's
        % rect, the dimension-B texture (stored after A's two) into B's
        Screen('DrawTextures', screen.win, [stim.tex(opt(1)), stim.tex(2 + opt(2))], [], stim.rects);

    otherwise
        error('drawTextures: unknown texture_name "%s".', texture_name)
end
end
