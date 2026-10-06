% Color channel experiment using FPVS
%   ----
% A checkerboard stimulus flickers at a specified frequency to drive a
% baseline SSVEP response. The color of the checkerboard changes at a lower
% frequency, driving a secondary SSVEP response that indexes the similarity
% between the baseline and "difference" color.
% - Angus Chapman, Spring 2022

% Yong comes in here! 6/8/2022
%   - 6Hz and 2Hz
%   - Make it even bigger!
%   - Instead of random colors, make it set colors.

% Yong comes in here again! 9/29/2022
%   - Make a fixation cross task (random fixation change at any point-up to five times?).


clear all
close all
clc

addpath(genpath('../../general scripts'));

Screen('Preference', 'SkipSyncTests', 1);

% Prefs and initialize
prefs = Preferences();
prefs = Initialize(prefs);

win = Screen('OpenWindow', prefs.monitor, [0 0 0]);
Screen('FillRect', win, prefs.backColor);
Screen('Flip', win);
Screen('TextSize', win, 30);
HideCursor();

Screen('BlendFunction', win, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

sid = prefs.sid;

eeg = 0; % set to 1 if you are sending triggers

% Information about the screen
rect = Screen('Rect', prefs.monitor);
[cx, cy] = RectCenter(rect); % get center of the screen

% Response buttons
KbName('UnifyKeyNames');
expAdv = KbName('c');

ListenChar(2); % suppress input

if eeg, [obj,port] = initializePort(166); end %start exp

% -------------------------------------------------------------------------
% Stimulus preparation
% -------------------------------------------------------------------------

% make sure colors are correctly presented on the monitor
if prefs.adjClut, originalClut = adjustMonitor(win); end

labwheel = load('colorwheelLAB.mat');
lum = 50; % CIE L* value

labvals = [repmat(lum,[360 1]) labwheel.p.cols_full_wheel];
rgbwheel = 255*lab2rgb(labvals,'WhitePoint','d50');

% cue
cuesizeAngle = .4; % degrees color cue covers
cueframeAngle = 2; % degrees of blank space around the color cue

cuesizePix = angle2pix(prefs, cuesizeAngle);
cuepos  = [cx - cuesizePix/2, cy - cuesizePix/2, cx + cuesizePix/2, cy + cuesizePix/2];

% Tagging stimulus details
trialLen = 20; % seconds
trialBL = 2; % padding at start and end of trial
tag.nFrames = secs2frames(prefs,trialLen+trialBL*2);
tag.mainFrames = secs2frames(prefs,trialLen);
tag.BLFrames = secs2frames(prefs,trialBL);

tag.wedges = 6; % number of checkerboard wedges
tag.angle = 360 ./ tag.wedges; % angle covered by each wedge
tag.nRings = 7; % number of concentric circles within a wedge (even number)
tag.maxRad = 28; %22;%16; % radial extend of tag stim

tag.rBound = linspace(cueframeAngle, tag.maxRad, tag.nRings+1); % space rings evenly
tag.rs = angle2pix(prefs,tag.rBound(2:end)); % outer radius of rings
tag.r0 = angle2pix(prefs,tag.rBound(1)); % inner radius of smallest ring

tag.blank = prefs.backColor;

% -- checkerboard flicker frequencies --
% make sure to choose these such that they evenly divide the frame rate of
% the monitor you are using to present stimuli (e.g., at 60 Hz you can
% flicker the checkerboard at 10 Hz and modulate the color at 1.33 Hz, but
% not at 3.5 Hz)

% baseline freq
tag.freq0 = 12; %10; % Hz
tag.period0 = prefs.frameRate ./ tag.freq0; % frames/flip
% "difference" freq
tag.freq1 = 2; %4/3;
tag.period1 = prefs.frameRate ./ tag.freq1;

% rects for arcs to appear in
rect0 = [cx-tag.r0/2, cy-tag.r0/2, cx+tag.r0/2, cy+tag.r0/2]; % donut hole
rectMat = zeros(numel(tag.rs), 4); % preset rectangle
for i=1:numel(tag.rs)
    rad = tag.rs(i);
    rectMat(i,:) = [cx-rad/2, cy-rad/2, cx+rad/2, cy+rad/2]; % box for ring
end

% angles to draw color within
angles = 0:tag.angle:(360-tag.angle);

% draw checkerboard stimulus
xylim = pi*tag.nRings;
[x,y] = meshgrid(-xylim : 2*xylim/(angle2pix(prefs,tag.maxRad)) : xylim, ...
    -xylim : 2*xylim/(angle2pix(prefs,tag.maxRad)) : xylim);
at = atan2(y,x);
checks = (1 + sign(sin(at * tag.wedges) + eps) ...
    .* sign(sin(sqrt(x.^2 + y.^2))))/2;
circle = x.^2 + y.^2 <= xylim^2 & x.^2 + y.^2 > 10;
checksA = circle .* checks;
checksB = circle .* (1-checks);

cbBaseA = repmat(checksA,[1 1 3]);
cbBaseB = repmat(checksB,[1 1 3]);

colorSimilarity = [15; 30; 60; 180];
nReps = 15; % repetitions of each similarity condition

trialSim = Shuffle(repmat(1:length(colorSimilarity),[1 nReps]));
nTrials = length(trialSim);
%keyboard;
for ii=1:nTrials

    d.sid(ii) = sid;
    d.trialNum(ii) = ii;
    d.freqBL(ii) = tag.freq0;
    d.freqDiff(ii) = tag.freq1;
    d.colorSimilarity(ii) = colorSimilarity(trialSim(ii));

    % -----------------------------------------------------------------
    % flickering stimulus preparation
    % -----------------------------------------------------------------

    randAng = randi(360); % randomize baseline color
    randRot = (-1).^(randi(2)); % randomize direction of difference color
    colorBL = rgbwheel(randAng,:);
    d.colorAngBL(ii) = randAng;
    d.colorBL(ii,:) = colorBL;

    % get "difference" color
    diffAng = randAng + randRot*colorSimilarity(trialSim(ii));
    if diffAng>360
        diffAng = diffAng-360;
    elseif diffAng<=0
        diffAng = diffAng+360;
    end
    colorDiff = rgbwheel(diffAng,:);
    d.colorDiffAng(ii) = diffAng;
    d.colorDiff(ii,:) = colorDiff;
    
    % make colored checkerboards
    thisCB(1,1) = Screen('MakeTexture', win, cbBaseA .* shiftdim(colorBL,-1));
    thisCB(1,2) = Screen('MakeTexture', win, cbBaseB .* shiftdim(colorBL,-1));
    thisCB(2,1) = Screen('MakeTexture', win, cbBaseA .* shiftdim(colorDiff,-1));
    thisCB(2,2) = Screen('MakeTexture', win, cbBaseB .* shiftdim(colorDiff,-1));

    % set up colors for checkerboard mask
    colMat = ones(1,tag.nFrames);
    colSwaps = find(mod(0:tag.nFrames-1, tag.period0*2)==0);
    for i=1:length(colSwaps)
        colMat(colSwaps(i):(colSwaps(i)+tag.period0)) = 2;
    end

    rgbMat = ones(1,tag.nFrames);
    rgbSwaps = find(mod(0:tag.nFrames-1, tag.period1*2)==0);
    rgbSwaps = rgbSwaps(rgbSwaps>tag.BLFrames);
    rgbSwaps = rgbSwaps(rgbSwaps<tag.mainFrames+tag.BLFrames);
    for i=1:length(rgbSwaps)
        rgbMat(rgbSwaps(i):(rgbSwaps(i)+tag.period0*2)) = 2;
    end
    
    %Fixation Task Set Up
    fixMat = zeros(1,tag.nFrames);
    numChange = randi(3,1);
    data.numChange(ii) = numChange;
    howlongfix = secs2frames(prefs, 0.3);
    minimumdifference = howlongfix*3;
    fixstart = randomwithrange(tag.BLFrames+howlongfix*2, tag.nFrames-tag.BLFrames-howlongfix*2, numChange, minimumdifference);
    for i = 1:numChange
        fixMat(fixstart(i):(fixstart(i)+howlongfix)) = 1;
    end
    
    % -----------------------------------------------------------------
    % show trial ------------------------------------------------------
    % -----------------------------------------------------------------

    % report similarity
    trText = 'Press any key to initiate the next trial';
    Screen('FillRect', win, prefs.backColor, rect);
    Screen('FillRect', win, prefs.white, cuepos);
    DrawFormattedText(win, trText, 'center', cy - 2*cuesizePix, prefs.white);
    Screen('Flip', win);
    KbWait();
    WaitSecs(1.0);

    % show initial screen
    Screen('FillRect', win, prefs.backColor, rect);
    Screen('FillRect', win, prefs.white, cuepos);
    Screen('Flip', win);
    if eeg, sendEventCode(obj,port,2); end % trial start

    trialJitter = randi([400 800])/1000; % random time 400-800 ms
    WaitSecs(trialJitter);

    frmCnt = 1;
    while frmCnt <= tag.nFrames
      % draw flickering stim
      Screen('DrawTexture', win, thisCB(rgbMat(frmCnt),colMat(frmCnt)));

      % fixation cue
      if fixMat(frmCnt) == 0
        Screen('FillRect', win, prefs.white, cuepos);
      elseif fixMat(frmCnt) == 1
        Screen('FillRect', win, prefs.gray, cuepos);
      end

      % draw stim on screen
      Screen('Flip', win);

      % EEG triggers as needed
      if eeg
          % trigger first checkerboard stim
          if frmCnt == 1
              sendEventCode(obj,port,4);
          % trigger main trial period start, dependent on similarity
          elseif frmCnt == tag.BLFrames
              sendEventCode(obj,port,4+trialSim(ii));
          % trigger end of main trial period
          elseif frmCnt == tag.BLFrames+tag.mainFrames
              sendEventCode(obj,port,16);
          end
      end

      % advance frame and flip colors when necessary
      frmCnt = frmCnt + 1;

    end

    Screen('FillRect', win, prefs.backColor, rect);
    Screen('Flip', win);
    if eeg, sendEventCode(obj,port,32); end % end trial
    WaitSecs(2.0);

    % report fixation change
    trText = 'How many times did it change? 1,2,3';
    Screen('FillRect', win, prefs.backColor, rect);
    Screen('FillRect', win, prefs.white, cuepos);
    DrawFormattedText(win, trText, 'center', cy - 2*cuesizePix, prefs.white);
    Screen('Flip', win);
    response = '';
    while isempty(response)
         [~,~,keysDown]=KbCheck;
         if keysDown(KbName('1!'))
             response='1';
         end
         if keysDown(KbName('2@'))
             response='2';
         end
         if keysDown(KbName('3#'))
             response='3';
         end
         if keysDown(KbName('q'))
             sca; error('User quit!');
         end
    end
    data.response(ii) = response;
    WaitSecs(0.1);

    %Feedback
    if str2num(response) == numChange
        Screen('FillRect', win, [0 256 0], cuepos);
    else
        Screen('FillRect', win, [256 0 0], cuepos);
    end
    Screen('Flip', win);
    WaitSecs(0.3);
end

save(sprintf('data/CCfpvs_s%s.mat',sid));

endText = 'All done!';
Screen('FillRect', win, prefs.backColor, rect);
Screen('FillRect', win, prefs.white, cuepos);
DrawFormattedText(win, endText, 'center', cy - 2*cuesizePix, prefs.white);
Screen('Flip', win);
WaitSecs(3.0);

sca


