function va = pixel2va(gx, gy, tarx, tary, parameters, screen)
% created by Mrugank Dake, Dartmouth College (09/03/2025)
% Eccentricity, in dva, of point (gx, gy) relative to (tarx, tary), all in pixels.
dx = gx - tarx;    % in pix
dy = -(gy - tary); % in pix
dx_cm = dx.*screen.pixSize; % in cm
dy_cm = dy.*screen.pixSize; % in cm
r = sqrt(dx_cm.^2 + dy_cm.^2); % in cm

% compute visual angle
va = atan2d(r, parameters.viewingDistance);
end
