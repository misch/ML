function Projected = project_MicheleWyss(Data, Basis)
% Input 
%     Data: the data matrix, each column represents a data point.
%     Basis: the PCA basis.
%
% Output
%     Projected: the projected data.
%

%assert(size(Data,2),size(Basis,1));
%assert(n>=k);
%% You know. The part with wrapping the head around the dimensions...
Data = Data';
Basis = Basis';

%% Simply project the data onto the new (smaller) basis
%  This is the key point of PCA because it will decrease the
%  dimensionality of the data (since we are going to have a 
%  representation of it in a lower dimensional space than before
size(Basis)
size(Data)
Projected = (Basis*Data)';