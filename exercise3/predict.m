function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add column of ones for X:
X = [ones(m, 1) X];

% putting in additional steps just for clarification and to match my notes.
z2=X*Theta1';

a2=sigmoid(z2);

% when computing a2, add column of ones for a2, the "bias term":

a2=[ones(m,1), a2];

% Now ready to compute the output layer
z3=a2*Theta2';

% Outputs, which are the h_theta(x) for each unit
a3=sigmoid(z3);

% Now, just as in predictOneVsAll, we need to pick out which element of each row
% vector is largest; this corresponds to the class predicted by the model

[amax,index]=max(a3,[],2);

p=index;


% =========================================================================


end
