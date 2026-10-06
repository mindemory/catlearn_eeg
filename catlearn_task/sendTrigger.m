function trigTime = sendTrigger(parameters, code)
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
%
% Single choke point for EEG triggers. Always returns the GetSecs timestamp
% at which the trigger was (or would have been) sent, so the main task can
% log event times whether or not EEG is actually being recorded.
%
% While parameters.EEG == 0 this is a no-op apart from the timestamp. To go
% live: set parameters.EEG = 1 and parameters.triggerMethod in
% loadParameters, then fill in the matching branch below. validateTriggers
% is called once at task startup so a misconfiguration fails before the
% first trial rather than mid-block.
%
% Trigger codes used by catlearn_task:
%   0   block start
%   1   fixation onset
%   21-24 stimulus (flicker) onset, compound 1-4 (= 2*levelA + levelB + 1)
%   3   stimulus offset = response window onset
%   41  response: F        42  response: J
%   51  feedback: correct  52  feedback: incorrect  53  feedback: timeout
%   6   ITI onset
%   7   block end

trigTime = GetSecs;
if ~parameters.EEG
    return;
end

switch parameters.triggerMethod
    case 'parallel'
        % Example (Linux/Windows, PTB IOPort or io64):
        %   io64(parameters.ioObj, parameters.triggerAddr, code);
        %   WaitSecs(0.005);
        %   io64(parameters.ioObj, parameters.triggerAddr, 0);
        error('sendTrigger: parallel-port branch not implemented yet (code %d).', code)

    case 'serial'
        % Example (BrainProducts TriggerBox / MarkStim over serial):
        %   IOPort('Write', parameters.triggerHandle, uint8(code), 0);
        error('sendTrigger: serial branch not implemented yet (code %d).', code)

    case 'python'
        % The srh_ori approach: shell out to a per-code python script.
        % Slow (tens of ms) and jittery -- prefer parallel/serial for EEG.
        %   system(sprintf('sudo python3 %s/eegflag%d.py', parameters.triggerPath, code));
        error('sendTrigger: python branch not implemented yet (code %d).', code)

    otherwise
        error('sendTrigger: unknown triggerMethod "%s".', parameters.triggerMethod)
end
end
