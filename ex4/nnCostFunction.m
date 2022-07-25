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
triangle1 = 0;
triangle2 = 0;

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

%add a column of 1 to X
A = ones(m);
B = eye(num_labels);
one = A(:,1);
X = [one,X];
%now X is m * n+1 i.e.401*25.

%1forewardprop
%2compute cost
for i = 1:m
    a1 = X(i,:)';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    ot = B(:,y(i));
    temp = sum((-ot) .* log(a3) - (1-ot) .* log(1-a3));
    J = J + temp;
    
    
    %backprop
    delta3 = a3 - ot;
    delta2 = (Theta2') * delta3 ;
    %now delta2 is 26 * 1.
    delta2 = delta2(2:end);
    %now delta2 is 25 * 1. The bias term is removed.
    delta2 = delta2 .* sigmoidGradient(z2)
    
    triangle1 = triangle1 + delta2 * a1';
    triangle2 = triangle2 + delta3 * a2';
    
end
J = J/m;
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
Theta1x = Theta1(:,2:end);
Theta2x = Theta2(:,2:end);
Theta1m = Theta1x.*Theta1x;
Theta2m = Theta2x.*Theta2x;
reg = (lambda / (2*m)) * (sum(Theta1m(:)) + sum(Theta2m(:)));
J = J + reg;


Theta1_grad = (1/m) * triangle1;
Theta2_grad = (1/m) * triangle2;


% -------------------------------------------------------------

%regularized nn -> compute theta derivative.
%diffrentiate row(i) or column(j)!! in regularization.

Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;

Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;

Theta1_grad = Theta1_grad + Theta1_reg * (lambda/m);
Theta2_grad = Theta2_grad + Theta2_reg * (lambda/m);


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
