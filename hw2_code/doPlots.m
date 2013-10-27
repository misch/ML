load('C:\Users\wyss\ML\hw2_code\plotVars1_90percent.mat')
load('C:\Users\wyss\ML\hw2_code\plotVars2_90percent.mat')

figure;
hold on;
plot(falseNegRate1, truePosRate1, 'b')
plot(falseNegRate2, truePosRate2, 'r')
legend('Log. Regression','GDA','Location','SouthEast')
axis( [0 1 0 1] );
title('ROC curve');
xlabel('False Positive Rate');
ylabel('True Positive Rate');


figure;
hold on;
plot(recall1,precision1,'b')
plot(recall2,precision2,'r')
axis( [0 1 0 1] );
title('precision-recall curve')
legend('Log. Regression', 'GDA','Location','SouthEast')
xlabel('Recall');
ylabel('Precision');