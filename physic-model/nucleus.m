function [phi,tempr] = nucleus(Nx,Ny,seed)
format long;

%for i=1:Nx
%for j=1:Ny

%phi(i,j) = 0.0;
%tempr(i,j) = 0.0;

%end
%end 

phi = zeros(Nx,Ny);
tempr = zeros(Nx,Ny);

for i=1:Nx
    for j=1:Ny
        if ((i-Nx/2)*(i-Nx/2)+(j-Ny/2)*(j-Ny/2) < seed)
            phi(i,j) = 1.0;
        end
    end
end

end %endfunction