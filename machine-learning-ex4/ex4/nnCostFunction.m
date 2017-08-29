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

	% size(X);
	% size(Theta1)  25 X 401
	a = ones([m; 1]);

	x1 = [a, X];

	z2 = x1*transpose(Theta1); % 5000 x 25

	x2 = z2;
	x2 = sigmoid(z2);

	x2 = [a, x2]; 

	z3 = x2*transpose(Theta2); %
	h = z3;
	h = sigmoid(z3);



	y1 = zeros(m, num_labels);

	for i = 1:m
		y1(i, y(i)) = 1;
	end

	yd = y1;
	y1 = y1(:);

	
	h1 = h(:);	

	J = (-(transpose(y1))*log(h1) - (1 - transpose(y1))*log(1 - h1))/m;

	%removing bais theta form Theta
	t1 = Theta1(1:hidden_layer_size, 2: input_layer_size+1);
	t2 = Theta2(1:num_labels,2:hidden_layer_size + 1);

	t1 = t1(:);
	t2 = t2(:);


	t1 = t1'*t1; %'
	t2 = t2'*t2; %'


	RF = lambda*(t1 + t2)/(2*m);

	J = J + RF;


	%%%%%%%%%%%%% next Setup
	a1 = x1;
	a2 = x2;
	a3 = h;
	size(yd); %5000 X 10
	size(a2); %5000 X 26
	size(a3); %5000 X 10
	size(z2); %5000 X 25
	size(z3); %5000 X 10
	size(Theta1); %25 X 401
	size(Theta2); %10 X 26

	gz3d = sigmoidGradient(z3);

	size(gz3d); % 5000 X 10
	del3 = a3 - yd; %5000 X 10
	temp = del3*Theta2(1:end, 2:end); % 5000 X 25
	%temp = temp(1:end, 2:end);


	RF1 = lambda*Theta1/m;
	RF2 = lambda*Theta2/m;	


	del2 = temp.*sigmoidGradient(z2);  %5000 X 26

	size(del2); % 5000 X 26
	d2 = (transpose(del3)*a2)/m;

	Theta2_grad = d2 + RF2;

	d1 = (transpose(del2)*a1)/m;

	Theta1_grad = d1 + RF1;


	size(d1);
	size(d2);


	Theta1_grad(1:end, 1) = d1(1:end, 1);
	Theta2_grad(1:end, 1) = d2(1:end, 1);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
