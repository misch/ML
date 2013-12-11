function Eigen = pca_MicheleWyss(Data, k)
% Input
%     Data: the data matrix, each column represents a data point.
%     k: number of principle components to use.
%
% Output
%     Eigen: a matrix with k columns, where each column is an 
%                eigenvector of the covariance matrix of Data.
%

[U, S, V] = svd(X);

% Take the first k eigenvectors of the covariance matrix
% that are in the first k columns of the U-matrix of the svd
Eigen = U(:,1:k);