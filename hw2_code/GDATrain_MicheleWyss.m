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

%Sigma = eye(dim);
Sigma = zeros(dim);
for i = 1:m
    meanVec = mu0;
    if (LabelsTrain(i) > 0)
        meanVec = mu1;
    end   
    Sigma = Sigma + ((DataTrain(i,:)'-meanVec)*((DataTrain(i,:)'-meanVec)'));   
end
Sigma = Sigma/m;
end