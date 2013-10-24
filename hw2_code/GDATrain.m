function [phi, mu0, mu1, Sigma] = GDATrain( DataTrain, LabelsTrain )

dim = size(DataTrain,2);

% dummy
phi = 0.5;
Sigma = eye(dim);
mu0 = zeros(dim,1); mu0([230 421 167]) = 75;
mu1 = zeros(dim,1); mu1([200 456 322]) = 130;

end

