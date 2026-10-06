function showprompts(screen, prompt_name, num, doflip, color)
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
% All experimenter/subject-facing text in one place.
if nargin < 3; num = []; end
if nargin < 4 || isempty(doflip); doflip = 1; end
if nargin < 5 || isempty(color); color = screen.white; end

fontsize = 30;
switch prompt_name
    case 'WelcomeWindow'
        text = ['Two flickering patterns will appear on either side of the cross.\n\n' ...
                'Keep your eyes on the central cross at all times.\n\n' ...
                'Each pair of patterns belongs to one of two groups: F or J.\n' ...
                'When the patterns disappear, press  F  or  J  to say which group.\n' ...
                'You will be told after every trial whether you were right.\n\n' ...
                'Press SPACE to begin ...'];
        fontsize = 28;
    case 'BlockExists'
        text = ['Block ' num2str(num) ' already exists. Press SPACE to overwrite, or ESCAPE to quit.'];
    case 'BlockStart'
        text = ['Block ' num2str(num) '\n\n Get ready ...'];
    case 'Correct'
        text = 'Correct';
    case 'Incorrect'
        text = 'Incorrect';
    case 'Timeout'
        text = 'Too slow';
    case 'BlockEnd'
        text = ['End of block.\n\n Accuracy: ' num2str(round(num), '%i') '%'];
    case 'ContinueorEsc'
        text = ['End of block ' num2str(num) '. Press SPACE to continue, or ESCAPE to quit.'];
    case 'TrialPause'
        text = 'Experiment is paused! Press SPACE to resume.';
    case 'EndExperiment'
        text = 'Thank you for your participation!';
    otherwise
        error('showprompts: unknown prompt_name "%s".', prompt_name)
end

Screen('TextSize', screen.win, fontsize);
DrawFormattedText(screen.win, text, 'center', 'center', color, [], [], [], 1.4);
if doflip == 1
    Screen('Flip', screen.win);
end
end
