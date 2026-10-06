function r_pix = va2pixel(parameters, screen, phi)
% created by Mrugank Dake, Dartmouth College (09/03/2025)
% Returns the HALF-extent, in pixels, of an object subtending phi degrees of
% visual angle at the current viewing distance. So:
%   diameter of a phi-deg patch      == 2 * va2pixel(..., phi)
%   offset of a patch at ecc e deg   ==     va2pixel(..., 2*e)
rho = parameters.viewingDistance;
r_cm = rho * tand(phi/2);
r_pix = round(r_cm/screen.pixSize); % convert cm to pixels
end
