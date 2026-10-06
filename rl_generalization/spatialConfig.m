function [cells, name] = spatialConfig(config)
% Created by Mrugank Dake, Dartmouth College (09/24/2025)
%
% Grid cells of the two stimulus locations for spatial configuration 1-18.
% Transcribed from CONFIGS in task_design/plot_spatial_configs.py -- keep
% the two in sync. The screen is a 7x7 grid of cells with fixation in the
% center cell; cells are (row, col), 0-indexed from the top left, so
% fixation is (3, 3).
%
% Output:
%   cells  2x2, row 1 = [row col] of location A, row 2 = location B
%   name   e.g. 'horizontal, mid'
%
%   #   name                          A        B
%   1   horizontal, near              (3,2)    (3,4)
%   2   horizontal, mid               (3,1)    (3,5)
%   3   horizontal, periphery         (3,0)    (3,6)
%   4   vertical, near                (2,3)    (4,3)
%   5   vertical, mid                 (1,3)    (5,3)
%   6   vertical, periphery           (0,3)    (6,3)
%   7   upper quadrants, near         (2,2)    (2,4)
%   8   upper quadrants, mid          (1,1)    (1,5)
%   9   upper quadrants, periphery    (0,0)    (0,6)
%   10  lower quadrants, near         (4,2)    (4,4)
%   11  lower quadrants, mid          (5,1)    (5,5)
%   12  lower quadrants, periphery    (6,0)    (6,6)
%   13  diagonal \, near              (2,2)    (4,4)
%   14  diagonal \, mid               (1,1)    (5,5)
%   15  diagonal \, periphery         (0,0)    (6,6)
%   16  diagonal /, near              (4,2)    (2,4)
%   17  diagonal /, mid               (5,1)    (1,5)
%   18  diagonal /, periphery         (6,0)    (0,6)
configs = {
    'horizontal, near',           [3 2; 3 4]
    'horizontal, mid',            [3 1; 3 5]
    'horizontal, periphery',      [3 0; 3 6]
    'vertical, near',             [2 3; 4 3]
    'vertical, mid',              [1 3; 5 3]
    'vertical, periphery',        [0 3; 6 3]
    'upper quadrants, near',      [2 2; 2 4]
    'upper quadrants, mid',       [1 1; 1 5]
    'upper quadrants, periphery', [0 0; 0 6]
    'lower quadrants, near',      [4 2; 4 4]
    'lower quadrants, mid',       [5 1; 5 5]
    'lower quadrants, periphery', [6 0; 6 6]
    'diagonal \, near',           [2 2; 4 4]
    'diagonal \, mid',            [1 1; 5 5]
    'diagonal \, periphery',      [0 0; 6 6]
    'diagonal /, near',           [4 2; 2 4]
    'diagonal /, mid',            [5 1; 1 5]
    'diagonal /, periphery',      [6 0; 0 6]
    };

if ~isscalar(config) || ~isnumeric(config) || config ~= round(config) ...
        || config < 1 || config > size(configs, 1)
    error('spatialConfig: config must be an integer from 1 to %d.', size(configs, 1))
end
name = configs{config, 1};
cells = configs{config, 2};
end
