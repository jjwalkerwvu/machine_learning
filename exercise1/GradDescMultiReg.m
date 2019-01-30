function [theta, J_history]=GradDescMultiReg(X,y,theta,reg_param,alpha,num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% normalize features?


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



  theta_old=theta;
  % employ the lrCostFunction to determine cost (J) and gradient (grad)
  [J,grad]=lrCostFunction(theta_old,X,y,reg_param);
  
  
  
  theta=theta-alpha*grad;
  
  % make an array of the deltas, to generalize this scheme for larger sizes of n
  % (number of features)
  %delta=1/m*(X*theta_old-y)'*X;
  %theta=theta_old-alpha*delta';
  





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    disp(num2str(J_history(iter)));
end

end
