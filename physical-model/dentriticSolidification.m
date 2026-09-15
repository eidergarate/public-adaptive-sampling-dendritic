function [phi, tempr, computeTime] = dendriticSolidification(testNumber, tau, epsilonb, kappa, delta, aniso, alpha, gamma, theta0, Nx, Ny, dx, dy, writeVTK, idxToSave, data_raw_path)
%== get initial wall time:
time0=clock();
format long;

%-- Simulation cell parameters:

NxNy= Nx*Ny;

%--- Time integration parameters:

totalSimTime = 0.4;
nstep  =  4000;
nprint =    50;
dtime  = 1e-4; 

%--- Courant / stability check for a explicit scheme:
%    Here we do a 2D criterion: CFL = 4 * kappa * (dtime / dx^2).
%    If CFL > 1, you often risk instability or blow-up.
cflNumber = 4 * kappa * dtime / (dx^2); % Consider that dx = dy, otherwise modify
delta = 2*1e-5;
dtimeNeeded = (1- delta)*(dx^2) / (4 * kappa);  % max stable timestep for CFL <= 1
if cflNumber > 1
    %warning('CFL = %.3f > 1 => scheme may blow up. Use dtime <= %.3g', ...
             %cflNumber, dtimeNeeded);
    dtime = dtimeNeeded; % Set time step to comply with stability criterion
    nstep = round (totalSimTime/dtime);
else
    %fprintf('CFL = %.3f <= 1: Condition satisfied.\n', cflNumber);
end


%disp(['Using new time step value: ', num2str(dtime)]);
%disp(['Number of time steps: ', num2str(nstep)]);

%--- Material specific parameters:

teq = 1.0;
seed = 5.0;
%
pix=4.0*atan(1.0);

% step scaling
original_nsteps = 4000;  


scaling_factor = nstep / original_nsteps;
new_idxToSave = round(idxToSave * scaling_factor);
new_idxToSave(new_idxToSave > nstep) = nstep;
boolVector = false(nstep, 1);
boolVector(new_idxToSave) = true;

%--- Initiliaze and introduce initial nuclai:

[phi,tempr] = nucleus(Nx,Ny,seed);

%--- Laplacian

r=zeros(1,Nx);
r(1:2)=[2,-1];
T=toeplitz(r);

E=speye(Nx);

grad=-(kron(T,E)+kron(E,T));

%-- for periodic boundaries

for i=1:Nx
ii=(i-1)*Nx+1;
jj=ii+Nx-1;
grad(ii,jj)=1.0;
grad(jj,ii)=1.0;

kk=NxNy-Nx+i;
grad(i,kk)=1.0;
grad(kk,i)=1.0;
end

laplacian = grad /(dx*dy);

%---
%--- Evolution
%---

%Create folder for storing results
str1 = 'Ref';
str2 = num2str(testNumber);
pathName = fullfile(data_raw_path, strcat(str1, '_', str2));
mkdir(pathName)

for istep =1:nstep

phiold =phi;

%---
% calculate the laplacians and epsilon:
%---

phi2 = reshape(phi',NxNy,1);

lap_phi2 = laplacian*phi2;

lap_phi2 = lap_phi2.';

lap_phi2 = lap_phi2(:);

R = ceil (length (lap_phi2) / Nx);

lap_phi = reshape (lap_phi2, Nx, R).';

%--

tempx = reshape(tempr',NxNy,1);

lap_tempx =laplacian*tempx;

lap_tempx = lap_tempx.';

lap_tempx = lap_tempx(:);

R2 = ceil (length (lap_tempx) / Nx);

lap_tempr = reshape (lap_tempx, Nx, R2).';

%--gradients of phi:

[phidy,phidx]=gradient_mat(phi,Nx,Ny,dx,dy);

%-- calculate angle:

theta =atan2(phidy,phidx);

%--- epsilon and its derivative:

epsilon = epsilonb*(1.0+delta*cos(aniso*(theta-theta0)));

epsilon_deriv = -epsilonb*aniso*delta*sin(aniso.*(theta-theta0));

%--- first term:

dummyx =epsilon.*epsilon_deriv.*phidx;

[term1,dummy] =gradient_mat(dummyx,Nx,Ny,dx,dy);

%--- second term:

dummyy =-epsilon.*epsilon_deriv.*phidy;

[dummy,term2] =gradient_mat(dummyy,Nx,Ny,dx,dy);

%--- factor m:

m =(alpha/pix)*atan(gamma*(teq-tempr));

%-- Time integration:

phi = phi +(dtime/tau) *(term1 +term2 + epsilon.^2 .* lap_phi + ...
			 phiold.*(1.0-phiold).*(phiold - 0.5 + m));


%-- evolve temperature:

tempr = tempr + dtime*lap_tempr + kappa*(phi-phiold);

% =========================== //
% === DIAGNOSTIC CHECKS ==== //
% =========================== //

printDiag = false;

% 1) Check for NaNs or Infs
if any(isnan(phi(:))) || any(isnan(tempr(:)))
    error('NaN detected in phi or tempr at iteration %d!', istep);
end
if any(isinf(phi(:))) || any(isinf(tempr(:)))
    error('Inf detected in phi or tempr at iteration %d!', istep);
end

% 2) Monitor min/max to see if values are exploding
maxPhi = max(phi(:));
minPhi = min(phi(:));
maxTempr = max(tempr(:));
minTempr = min(tempr(:));

if printDiag
    % Print or log occasionally (e.g., every 50 steps) so it doesn’t clutter
    if mod(istep, 50) == 0
        %fprintf('Step %d: phi in [%.3g, %.3g], tempr in [%.3g, %.3g]\n', ...
            %istep, minPhi, maxPhi, minTempr, maxTempr);
    end

end

%-- write to VTK depending on the flag

if ((writeVTK == true) && (mod(istep,nprint) == 0 ))
cd (pathName)
fname=sprintf('time_%d.vtk',istep);
out =fopen(fname,'w');

nz=1;

npoin =Nx*Ny*nz;

% start writing ASCII VTK file:

% header of VTK file

fprintf(out,'# vtk DataFile Version 2.0\n');
fprintf(out,'time_10.vtk\n');
fprintf(out,'ASCII\n');
fprintf(out,'DATASET STRUCTURED_GRID\n');

%--- coords of grid points:

fprintf(out,'DIMENSIONS %5d  %5d  %5d\n',Nx,Ny,nz);
fprintf(out,'POINTS %7d   float\n',npoin);

optPerm = true;

if optPerm
    % Vectorized approach (single call):
    % Create meshgrid of coordinates
    [X, Y] = meshgrid(0 : dx : (Nx-1)*dx, 0 : dy : (Ny-1)*dy);

    % Z is all zeros in your code
    Z = zeros(size(X));

    % coords becomes 3 x (Nx*Ny), column-major order
    coords = [X(:), Y(:), Z(:)].';
    % Single fprintf call
    fprintf(out, '%14.6e   %14.6e   %14.6e\n', coords);

else
    % Original loop-based approach
    for i = 1:Nx
        for j = 1:Ny
            x = (i-1)*dx;
            y = (j-1)*dy;
            z = 0.0;
            fprintf(out, '%14.6e   %14.6e   %14.6e\n', x, y, z);
        end
    end
end


%--- write grid point values:

fprintf(out,'POINT_DATA %5d\n',npoin);

fprintf(out,'SCALARS OP  float  1\n');

fprintf(out,'LOOKUP_TABLE default\n');


% --- Writing phi
if optPerm
    fprintf(out, '%14.6e\n', phi(:));
else
    for i = 1:Nx
        for j = 1:Ny
            ii=(i-1)*Nx+j;
            fprintf(out,'%14.6e\n',phi(i,j));
        end
    end
end

fprintf(out,'SCALARS Tempr  float  1\n');

fprintf(out,'LOOKUP_TABLE default\n');

% --- Writing tempr
if optPerm
    % Vectorized approach: writes tempr in one shot
    fprintf(out, '%14.6e\n', tempr(:));
else
    % Loop-based approach: writes tempr element by element
    for i = 1:Nx
        for j = 1:Ny
            fprintf(out,'%14.6e\n', tempr(i,j));
        end
    end
end

fclose(out);
cd ..
end

%-- save results to folder

if boolVector(istep) == true

    adjusted_index = find(new_idxToSave == istep, 1);  % Buscar el índice en new_idxToSave
    if isempty(adjusted_index)
      warning('No se encontró un índice original para el índice ajustado %d', istep);
      continue;
    end
    original_index = idxToSave(adjusted_index);  % Obtener el índice original


    opName = fullfile(pathName, strcat(str1, '_', str2, '_op_', num2str(original_index), '.csv'));
    tempName = fullfile(pathName, strcat(str1, '_', str2, '_temp_', num2str(original_index), '.csv'));

    dlmwrite(opName, phi, 'delimiter', ',');
    dlmwrite(tempName, tempr, 'delimiter', ',');
end

end %istep


%--- calculate compute time:
computeTime = etime(clock(),time0);



