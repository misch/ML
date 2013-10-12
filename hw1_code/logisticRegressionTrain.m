function [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, maxIterations )
% logisticRegressionTrain train a logistic regression classifier
% [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, MaxIterations )
% Using the training data in DataTrain and LabelsTrain trains a logistic
% regression classifier theta. 
% 
% Implement a Newton-Raphson algorithm.

% dummy
dim = size(DataTrain,2);
theta = zeros(dim,1);
theta(250) = -1/dim;
theta(300) = 1/dim;

end