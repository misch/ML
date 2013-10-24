function [phi, mu0, mu1, Sigma] = GDATrain_MicheleWyss( DataTrain, LabelsTrain )
% Author: Michèle Wyss
% Immatriculation No.: 10-104-123

dim = size(DataTrain,2);

% map labels to {0,1}
LabelsTrain = LabelsTrain>0;

% phi
phi = sum(LabelsTrain)/length(LabelsTrain);

%%Compute the mean vectors - sketch of what I'm doing
%
%
%                                                       3 4 5 6
% given the labels e.g. [0 1 1] and the training data   4 5 6 7 , 
%                                                       5 6 7 8
% I just replicate the labels vector and transpose it:
%
%   0 0 0 0 
%   1 1 1 1
%   1 1 1 1
%
% After multiplying this component-wise with the training data,
% the row_i will be zero where label_i is zero, and it will be the original
% row if label_i is one. I can then simply sum up the columns.
%
%   0 0 0 0     3 4 5 6     0 0 0 0
%   1 1 1 1 .*  4 5 6 7 =   4 5 6 7 ===> sum(...) = [9 11 13 15]
%   1 1 1 1     5 6 7 8     5 6 7 8
% 
% To get the indicator function 1{label = 0}, I take the negation (~) of the
% labels.

mu0 = sum(repmat(~LabelsTrain,dim,1)' .* DataTrain)/sum(~LabelsTrain);
mu0 = mu0';

mu1 = sum(repmat(LabelsTrain,dim,1)' .* DataTrain)/sum(LabelsTrain);
mu1 = mu1';

% dummy
% phi = 0.5;
Sigma = eye(dim);
for i = 1:dim  
   if(LabelsTrain(i) == 1)
       Sigma = Sigma + (DataTrain(i,:)'-mu1)*((DataTrain(i,:)'-mu1)');
   end
   
   if(LabelsTrain(i) == 0)
       Sigma = Sigma + (DataTrain(i,:)'-mu0)*((DataTrain(i,:)'-mu0)');
   end
end
Sigma = Sigma/dim;

end

