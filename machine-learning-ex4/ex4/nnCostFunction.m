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


%
% Part 1 - Feedforward
%

% append ones as first column of X
X = [ones(m, 1) X];

% output for each layer
a2 = sigmoid(Theta1 * X');
a2 = [ones(1, size(a2, 2)); a2];
a3 = sigmoid(Theta2 * a2);

% output
h_theta_X = a3; % would be size of 10 x 5000

%
% expand y (5000 x 1) to Y (5000 x 10)
% e.g.  [2 4 5 1 10 9 8 .....] becomes
%      [
%       [0 1 0 0 0 0 0 0 0  0]
%       [0 0 0 1 0 0 0 0 0  0]
%       [0 0 0 0 1 0 0 0 0  0]
%       [0 0 0 0 0 0 0 0 0 10]
%       [0 0 0 0 0 0 0 0 1  0]
%       [0 0 0 0 0 0 0 1 0  0]
%       ....
%      ]
Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1;
endfor

% now Y would be size of 10 x 5000 (as h_theta_X)
Y = Y';

% arrayfun works like a 'zipWith' function
costFuncForOneSampleOneLabel = @(h_theta_x, y) -y*log(h_theta_x) - (1-y)*log(1-h_theta_x);
J = (1/m) * sum(arrayfun(costFuncForOneSampleOneLabel, h_theta_X, Y)(:));

% regularized cost function
regularized_term_theta1 = sum(arrayfun(@(x)x^2, Theta1(:, 2:end))(:));
regularized_term_theta2 = sum(arrayfun(@(x)x^2, Theta2(:, 2:end))(:));
regularized_term = (lambda/(2*m)) * (regularized_term_theta1 + regularized_term_theta2);
J = J + regularized_term;

% -------------------------------------------------------------

%
% Part 2 - Backpropagation
%

for t = 1:m

    % step 1
    a_1 = X(t, :)';             % 401 x 1
    z_2 = Theta1 * a_1;         %  25 x 1
    a_2 = [1; sigmoid(z_2)];    %  26 x 1
    z_3 = Theta2 * a_2;         %  10 x 1
    a_3 = sigmoid(z_3);         %  10 x 1

    % step 2
    err_3 = a_3 - Y(:, t);  % 10 x 1

    % step 3
    err_2 = (Theta2' * err_3)(2:end) .* sigmoidGradient(z_2);   % 25 x 1

    % step 4
    Theta2_grad = Theta2_grad + err_3 * a_2';   % 10 x  26

    Theta1_grad = Theta1_grad + err_2 * a_1';   % 25 x 401

endfor

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% regularized theta grad
Theta1_grad = Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end




% Helper Function - expand y (5000 x 1) to (5000 x 10)
% e.g.  [2 4 5 1 10 9 8 .....] becomes
%      [
%       [0 1 0 0 0 0 0 0 0  0]
%       [0 0 0 1 0 0 0 0 0  0]
%       [0 0 0 0 1 0 0 0 0  0]
%       [0 0 0 0 0 0 0 0 0 10]
%       [0 0 0 0 0 0 0 0 1  0]
%       [0 0 0 0 0 0 0 1 0  0]
%       ....
%      ]
function y_vec = generateYMatrix(label, num_labels)
    y_vec = zeros(1, num_labels);
    y_vec(1, label) = 1;
end
