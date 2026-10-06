function [kbx, parameters] = initPeripherals(parameters)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
%
% Finds the keyboard and sets up a KbQueue restricted to the keys this task
% cares about. A restricted keylist means KbQueueCheck only ever reports
% F/J/space/escape, so stray keypresses cannot be mistaken for responses.
% Code is adapted for Mac and Ubuntu.
KbName('UnifyKeyNames');

% Keys of interest
parameters.resp_keys = {'f', 'j'};
parameters.space_key = 'space';
parameters.exit_key = 'ESCAPE';
keyNames = [parameters.resp_keys, {parameters.space_key, parameters.exit_key}];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Locate the keyboard
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters.kbName lets you pin a specific device on the recording rig
% (run GetKeyboardIndices to see the products PTB can see).
[kbIdx, products] = GetKeyboardIndices();
if isempty(kbIdx)
    error('initPeripherals: no keyboard found by GetKeyboardIndices.')
end

if isfield(parameters, 'kbName') && ~isempty(parameters.kbName)
    match = find(contains(products, parameters.kbName));
    if isempty(match)
        error('initPeripherals: no keyboard matching "%s". Available: %s', ...
            parameters.kbName, strjoin(products, ', '))
    end
    kbx = kbIdx(match(1));
else
    kbx = kbIdx(1);
end

% Report the choice. On Linux one physical keyboard usually shows up as
% several devices (the keyboard plus consumer/system-control nodes), and
% only one of them reports letter keys -- picking the wrong one looks like
% "the task ignores keypresses". If the device named below is not the one
% subjects will type on, set parameters.kbName to pin the right one.
fprintf('initPeripherals: %d keyboard device(s) found:\n', numel(kbIdx));
for ii = 1:numel(kbIdx)
    if kbIdx(ii) == kbx
        fprintf('  [%d] %s   <-- using this one\n', kbIdx(ii), products{ii});
    else
        fprintf('  [%d] %s\n', kbIdx(ii), products{ii});
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the queue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
keyList = zeros(1, 256);
keyList(KbName(keyNames)) = 1;
parameters.keyList = keyList;

KbQueueRelease(kbx); % harmless if no queue exists; avoids "queue already created" errors
KbQueueCreate(kbx, keyList);
end
