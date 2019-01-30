function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ====================== Part1 ======================
% calculate the cost function
% need to add an extra column to X (a bias term)
X = [ones(m, 1) X];
% set up logical matrix for y vector; place 1 into the column of the 
% row vector for example i at the row corresponding to the number of the 
% "output" case
ymat1=y*ones(1,num_labels);
ymat2=ones(m,1)*(1:num_labels);
%y_arr=zeros(m,num_labels);
y_class=ymat1==ymat2;
% we slice the mxnum_labels size array, then concatenate all of the rows 
% together
y_arr=(y_class'(:))';

% compute the cost

% perform the necessary matrix computations:
z2=X*Theta1';
a2=sigmoid(z2);
% when computing a2, add column of ones for a2, the "bias term":
a2=[ones(m,1), a2];
% Now ready to compute the output layer
z3=a2*Theta2';
% Outputs, which are the h_theta(x) for each unit
a3_temp=sigmoid(z3);
a3=(a3_temp'(:));

% I think this works for just the cost without regularization?
J_0=-1/m*(y_arr*log(a3)+(1-y_arr)*(log(1-a3)));

% the cost from the regularization term; make sure to not include the bias 
% terms! that's why we index (:,2:end).
J_reg=lambda/2/m*(sum(sum((Theta1(:,2:end).^2)))+sum(sum((Theta2(:,2:end).^2))));

J=J_0+J_reg;

% ====================== Part2 ======================
% perform backpropagation

% use a loop for now, since the pdf suggests doing this

% initialize gradient accumulator?
grad_accum1=0;
grad_accum2=0;

for t=1:m
  % step1: forward propagation: set the input layer's values to the t-th 
  % training example
  a1=X(t,:)';
  z2=Theta1*a1;
  % don't forget to add 1 to a2 array!
  a2=[1;sigmoid(z2)];
  z3=Theta2*a2;
  
  % step2:
  % take transpose of the argument below to make delta3 a column vector
  delta3=(a3_temp(t,:)-y_class(t,:))';
  
  % step3:
  % don't forget to add 1!
  delta2=Theta2'*delta3.*sigmoidGradient([1;z2]);
  
  % step 4:
  % the gradient accumulator? for each layer?
  grad_accum1=grad_accum1+delta2(2:end)*a1';
  grad_accum2=grad_accum2+delta3*a2';
  
  % unregularized version:
  %Theta1_grad=1/m*grad_accum1;
  %Theta2_grad=1/m*grad_accum2;
  
  % regularized version:
  reg_term1=lambda/m*Theta1(:,2:end);
  reg_term2=lambda/m*Theta2(:,2:end);
  Theta1_grad=1/m*grad_accum1+[zeros(size(Theta1,1), 1) reg_term1];
  Theta2_grad=1/m*grad_accum2+[zeros(size(Theta2,1), 1) reg_term2];
  
end




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
