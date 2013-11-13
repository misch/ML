% Load training data X and their labels y
vl_setup demo % to load the demo data
load('vl_demo_svm_data.mat');
X = [6 6; 6.5 6.5; 7 7; 1 1; 2 2; 1.5 1.5;   5 7]';
X = X - repmat(mean(X,2), [1, length(X)]);
y = [1 1 1 -1 -1 -1  -1];
Xp = X(:,y==1);
Xn = X(:,y==-1);

figure
plot(Xn(1,:),Xn(2,:),'*r')
hold on
plot(Xp(1,:),Xp(2,:),'*b')
axis equal ;

lambda = 10 ; % Regularization parameter
maxIter = 1000 ; % Maximum number of iterations


[w b info] = vl_svmtrain(X, y, lambda, 'MaxNumIterations', maxIter)


% Visualisation
eq = [num2str(w(1)) '*x+' num2str(w(2)) '*y+' num2str(b)];
line = ezplot(eq);
set(line, 'Color', [0 0.8 0],'linewidth', 2);