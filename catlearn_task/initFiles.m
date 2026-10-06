function parameters = initFiles(parameters, screen, data_path, kbx, block)
% Created by Mrugank Dake, Dartmouth College (09/03/2025)
% Creates the per-block output directory and fixes the output file names.
% If the block directory already exists the experimenter is asked to confirm
% (SPACE) or bail out (ESCAPE) before anything gets overwritten.
block_dir = [data_path filesep 'block' num2str(block, '%02d')];

if exist(data_path, 'dir') ~= 7
    mkdir(data_path);
end

if exist(block_dir, 'dir') == 7
    KbQueueStart(kbx);
    KbQueueFlush(kbx);
    spaceCode = KbName(parameters.space_key);
    exitCode = KbName(parameters.exit_key);
    confirmed = false;
    while ~confirmed
        showprompts(screen, 'BlockExists', block);
        [pressed, firstPress] = KbQueueCheck(kbx);
        if pressed
            if firstPress(exitCode) > 0
                KbQueueStop(kbx);
                error('initFiles: aborted by experimenter, block %02d already exists.', block)
            end
            confirmed = firstPress(spaceCode) > 0;
        end
    end
    KbQueueStop(kbx);
else
    mkdir(block_dir);
end

% File names that will be saved.
parameters.block = num2str(block, '%02d');
parameters.block_dir = block_dir;
parameters.matFile = ['matFile_subj' parameters.subject '_block' parameters.block '.mat'];
parameters.EEGreportFile = ['EEGreportFile_subj' parameters.subject '_block' parameters.block '.mat'];
end
