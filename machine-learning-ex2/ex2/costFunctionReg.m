function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

	m = length(X(:, 1));
	z = ones(m, 1);

	z = X*theta;

	h = sigmoid(z);
	h2 = 1 - h;

	y2 = 1 - y;

	l1 = log(h);
	l2 = log(h2);

	% J = (transpose(l1)*(-y) - transpose(l2)*y2)/m + (lambda/(2*m))* transpose(theta)*theta;
	theta;
	theta2 = transpose(theta)*theta;
	reg = (lambda* transpose(theta)*theta)/(2*m) - lambda*theta(1)*theta(1)/(2*m);
	J = (transpose(log(h))*(-y) - transpose(log(1-h))*(1 - y))/m + reg;

	h2 = h - y;

	grad = (transpose(X)*h2)/m +  (lambda/m)*theta;

	grad(1) = grad(1) - (lambda/m)*theta(1);
	% disp('hello');
% =============================================================

end
