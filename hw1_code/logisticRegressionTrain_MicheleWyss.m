function [ theta ] = logisticRegressionTrain_MicheleWyss( DataTrain, LabelsTrain, maxIterations )
% logisticRegressionTrain train a logistic regression classifier
% [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, MaxIterations )
% Using the training data in DataTrain and LabelsTrain trains a logistic
% regression classifier theta. 
% 
% Implement a Newton-Raphson algorithm.

% DataTrain = 
% x1.1 x1.2 x1.3 x1.4 ... x1.576
% ...
% xm.1 xm.2 x2.3 ...  ... xm.576

% LabelsTrain = 
% (y1 y2 y3 y4 ... ym)
%
%

% map labels to [0,1]
LabelsTrain = (LabelsTrain > 0);

% number of training samples
m = size(LabelsTrain,2);

theta = zeros(577,maxIterations);

% sigmoid function
sigmoid = @(x) 1/(1 + exp(-x));

% returns the value of h_theta(x).
% theta and x have to be column vectors.
h = @(theta_vec,x) sigmoid(theta_vec'*x);

%grad_log_likelihood = @(theta) 1/m * sum((LabelsTrain(1,:) - h(theta,DataTrain(:,:)')) * DataTrain(:,:));

%hessian = @(theta_vec) (-1/m) * sum(h(theta_vec,DataTrain(:,:)') .* (1 - h(theta_vec,DataTrain(:,:)')) .* DataTrain(:,:)' * DataTrain(:,:));

for i=2:maxIterations

    % compute gradient of the log likelihood
    grad_log_likelihood = zeros(1,577);
    for k=1:m
        grad_log_likelihood = grad_log_likelihood + (LabelsTrain(1,k) - h(theta(:,i-1),DataTrain(k,:)')) * DataTrain(k,:);
    end
    
    grad_log_likelihood = grad_log_likelihood/m;
    
    
    % compute hessian
    hessian = zeros(577);
    for k=1:m
        hessian = hessian + (h(theta(:,i-1),DataTrain(k,:)') .* (1 - h(theta(:,i-1),DataTrain(k,:)')) .* DataTrain(k,:)' * DataTrain(k,:));
    end
    
    hessian = hessian ./m;

    theta(:,i) = theta(:,i-1) - inv(hessian) * grad_log_likelihood';
end

theta = theta(:,maxIterations);
