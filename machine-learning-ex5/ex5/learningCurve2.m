function [error_train1av, error_val1av] = ...
    learningCurve2(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%
C=50;
M=10;
error_train1 = zeros(M, 1);
error_val1   = zeros(M, 1);
error_train1sum=zeros(M, 1);
error_val1sum=zeros(M, 1);
for z=1:C,
    k1 = randperm(12);
    k2 = randperm(21);
    Xrand = X(k1(1:M),:);
    XvalRand = Xval(k2(1:M),:);
    yrand = y(k1(1:M),:);
    yvalRand = yval(k2(1:M),:);
    
    for i = 1:M,
        theta=trainLinearReg(Xrand(1:i,:), yrand(1:i), lambda);
        error_train1(i)=linearRegCostFunction(Xrand(1:i,:), yrand(1:i), theta, 0);
        error_val1(i)=linearRegCostFunction(XvalRand, yvalRand, theta, 0);
    end
error_train1sum=error_train1sum+error_train1;
error_val1sum=error_val1sum+error_val1;
end
error_train1av=(1/C)*error_train1sum;
error_val1av=(1/C)*error_val1sum;


