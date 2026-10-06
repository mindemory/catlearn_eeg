function screen = initScreen(parameters)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
% Opens the PTB window and precomputes the pixel <-> dva conversion factors
% that va2pixel relies on.

% Make the window transparent so the MATLAB console stays visible while debugging
if parameters.isDemoMode
    PsychDebugWindowConfiguration(0, parameters.transparency);
end

screen.id = max(Screen('Screens')); % get the screen
[screen.screenXpixels, screen.screenYpixels] = Screen('WindowSize', screen.id); % x and y pixels of screen
[screen.screenWidth, screen.screenHeight] = Screen('DisplaySize', screen.id);   % screen width and height in mm
screen.screenWidth = screen.screenWidth/10;   % mm to cm
screen.screenHeight = screen.screenHeight/10; % mm to cm
pixWidth = screen.screenWidth/screen.screenXpixels;   % cm/pixel
pixHeight = screen.screenHeight/screen.screenYpixels; % cm/pixel
screen.pixSize = mean([pixWidth, pixHeight]);         % cm/pixel
screen.xCenter = screen.screenXpixels/2; % pixels
screen.yCenter = screen.screenYpixels/2; % pixels

screen.deg_width = atand(screen.screenWidth/2 / parameters.viewingDistance) * 2;   % width of screen in dva
screen.deg_height = atand(screen.screenHeight/2 / parameters.viewingDistance) * 2; % height of screen in dva

% Initialize colors
screen.white = WhiteIndex(screen.id);
screen.black = BlackIndex(screen.id);
screen.grey = screen.white*0.5;

% Initialize Screen Window
AssertOpenGL;
[screen.win, screen.screenRect] = PsychImaging('OpenWindow', screen.id, screen.grey, [], 32, 2, [], [], kPsychNeed32BPCFloat);
screen.ifi = Screen('GetFlipInterval', screen.win);
Screen('BlendFunction', screen.win, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Retrieve the maximum priority number
topPriorityLevel = MaxPriority(screen.win);
Priority(topPriorityLevel);
end
