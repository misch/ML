function [ theta ] = logisticRegressionTrain_MicheleWyss( DataTrain, LabelsTrain, maxIterations )
% Using the training data in DataTrain and LabelsTrain trains a logistic
% regression classifier theta. 
% 
% Implement a Newton-Raphson algorithm.
%
% DataTrain = 
% x1.1 x1.2 x1.3 x1.4 ... x1.576
% ...
% xm.1 xm.2 x2.3 ...  ... xm.576
%
% LabelsTrain = 
% (y1 y2 y3 y4 ... ym)

%% Initializations

% map labels to [0,1]
LabelsTrain = (LabelsTrain > 0);

% number of training samples
m = size(LabelsTrain,2);

% feature dimension
dimension = size(DataTrain,2);

theta = zeros(dimension,1);

% sigmoid function
sigmoid = @(x) 1/(1 + exp(-x));

% returns the value of h_theta(x).
% theta and x have to be column vectors.
h = @(theta_vec,x) sigmoid(theta_vec'*x);

%% Optimization
for i=1:maxIterations

    %% compute gradient of the log likelihood
    grad_log_likelihood = zeros(1,dimension);
    for k=1:m
        grad_log_likelihood = grad_log_likelihood + (LabelsTrain(k) - h(theta,DataTrain(k,:)')) * DataTrain(k,:);
    end
    
    grad_log_likelihood = grad_log_likelihood/m;
    
    
    %% compute hessian
    hessian = zeros(dimension);
    for k=1:m
        hessian = hessian + (h(theta,DataTrain(k,:)') .* (1 - h(theta,DataTrain(k,:)')) .* DataTrain(k,:)' * DataTrain(k,:));
    end
    
    hessian = -hessian ./m;
    
    %% update theta (Newton-Raphson)
    theta = theta - inv(hessian) * grad_log_likelihood';
end
