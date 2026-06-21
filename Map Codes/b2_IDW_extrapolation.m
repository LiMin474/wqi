function b2_IDW_extrapolation (wqdata)
explat = wqdata.Latitude; % Assuming latitude is in the first column
explon = wqdata.Longitude; % Assuming longitude is in the second column
%expc = expdata.NO3 ; % Assuming sample concentration is in the third column


GT = readgeotable("myshape1.shp");
jajpur = geotable2table(GT,["Lat","Lon"]);
jprLat=jajpur.Lat{:};
jprLon=jajpur.Lon{:};
[mod_Lat, mod_Lon] = meshgrid(linspace(min(jprLat), max(jprLat), 500), linspace(min(jprLon), max(jprLon), 500));

mod_conc=gridpoint(jprLat,jprLon,mod_Lat,mod_Lon,wqdata,explat,explon);


%save geoplott_3.mat jprLat jprLon mod_Lat mod_Lon mod_conc explat explon
%save geoplott.mat jprLat jprLon mod_Lat mod_Lon mod_conc explat explon %F, NO3
save geoplott_2.mat jprLat jprLon mod_Lat mod_Lon mod_conc explat explon %GQ,LQ,AQ,RQ

function projct=gridpoint(jprLat,jprLon,mod_Lat,mod_Lon,expconc,explat,explon)
%list={'Cl','SO4','PO4','U','CaH','MgH','HCO3'}; %% you can have water quality rating as well etc
%list={'F','NO3'};
list={'GQ','LQ','AQ','RQ'};

for j=1:size(list,2)
%intpconc = griddata(explat, explon, expconc.(list{j}), mod_Lat, mod_Lon, 'cubic');
Fint = idw1([explat explon],expconc.(list{j}),[reshape(mod_Lat,[],1) reshape(mod_Lon,[],1)]);
intpconc=reshape(Fint,[],length(mod_Lat));
inPoly = inpolygon(mod_Lat, mod_Lon, jprLat,jprLon);
intpconc(~inPoly) = NaN;
projct.(list{j})=intpconc;
end

function Fint = idw1(X0,F0,Xint,p,rad,L)
% function Fint = idw(X0,F0,Xint,p,rad,L)
%
% Inverse distance weight function to interpolate values based on
% sampled points.
%
% Fint = idw(X0,F0,Xint) uses input coordinates X0 and input values F0
% where X0 is a N by M input matrix of N samples and M number of variables.
% F0 is vector of N responses. Xint is a Q by M matrix of coordinates to be
% interpolated. Fint is the vector of Q interpolated values.
%
% Fint = idw(X0,F0,Xint,p,rad) uses the power p (default p = 2) and radius
% rad (default rad = inf).
%
% Fint = idw(X0,F0,Xint,p,rad,L) uses L-distance. By defaults L=2
% (Euclidean norm).
%
% Example:
%
% X1 = [800;2250;3250;2250;900;500];
% X2 = [3700;4200;5000;5700;5100;4900];
% F =  [13.84;12.15;12.87;12.68;14.41;14.59];
% Q = 100;
% [X1int,X2int] = meshgrid(0:4000/(Q-1):4000, 3200:(5700-3200)/(Q-1):5700);
% Fint = idw([X1,X2],F,[X1int(:),X2int(:)]);
% contourf(X1int, X2int, reshape(Fint,Q,Q), 20)
%
% Contact info:
%
% Andres Tovar
% tovara@iupui.edu
% Indiana University-Purdue University Indianapolis
%
% Code developed for the course Design of Complex Mechanical Systems (ME
% 597) offered for the first time in Spring 2014
% Default input parameters
if nargin < 6
    L = 2;
    if nargin < 5
        rad = inf;
        if nargin < 4
            p = 2;
        end
    end
end
% Basic dimensions
N = size(X0,1); % Number of samples
M = size(X0,2); % Number of variables
Q = size(Xint,1); % Number of interpolation points
% Inverse distance weight output
Fint = zeros(Q,1);
for ipos = 1:Q    
    % Distance matrix
    DeltaX = X0 - repmat(Xint(ipos,:),N,1);
    DabsL = zeros(size(DeltaX,1),1);
    for ncol = 1:M
        DabsL = DabsL + abs(DeltaX(:,ncol)).^L;
    end
    Dmat = DabsL.^(1/L);
    Dmat(Dmat==0) = eps;
    Dmat(Dmat>rad) = inf;
    
    % Weights
    W = 1./(Dmat.^p);
    
    % Interpolation
    Fint(ipos) = sum(W.*F0)/sum(W);
end
