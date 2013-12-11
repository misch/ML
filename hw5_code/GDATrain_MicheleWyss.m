function [phi, mu0, mu1, Sigma] = GDATrain_MicheleWyss( DataTrain, LabelsTrain )
% Author: Michèle Wyss
% Immatriculation No.: 10-104-123

dim = size(DataTrain,2);
m = length(LabelsTrain);
% map labels to {0,1}
LabelsTrain = LabelsTrain>0;

% phi
phi = sum(LabelsTrain)/m;

%%Compute the mean vectors - sketch of what I'm doing
%
%
% From DataTrain, select the rows where LabelsTrain is 1
% with DataTrain(LabelsTrain,:).
mu0 = (sum(DataTrain(~LabelsTrain,:))/sum(~LabelsTrain))';
mu1 = (sum(DataTrain(LabelsTrain,:))/sum(LabelsTrain))';

% setup a matrix meanMatrix where the i-th row contains mu0 if LabelsTrain(i) = 0
% and mu1 if LabelsTrain(i) = 1
%
% E.g.:					mu0(1) mu0(2) ... mu0(576)
% Labels = [0 1 0] ===> meanMatrix =	mu1(1) mu1(2) ... mu1(576)
%					mu0(1) mu0(2) ... mu0(576)
%
%
%

labelRep = repmat(LabelsTrain,[dim,1])';

meansMatrix = ((repmat(mu0,[1,m])' .* ~labelRep) + (repmat(mu1,[1,m])' .* labelRep));

Sigma = ((DataTrain - meansMatrix)' * (DataTrain - meansMatrix))/m;
end