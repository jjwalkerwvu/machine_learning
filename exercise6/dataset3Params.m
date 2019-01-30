function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% will be using these lines a lot; copied here for convenience
%C=1;
%sigma=0.1;
%model= svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%visualizeBoundary(Xval, yval, model);
% a line here to see how well the model is matching the prediction.
% compute the prediction.
%prediction = svmPredict(model, Xval);
%mean(double(prediction ~= yval))




% =========================================================================

end
