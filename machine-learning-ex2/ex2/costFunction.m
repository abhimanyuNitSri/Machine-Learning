function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

	m = length(X(:, 1));
	z = ones(m, 1);

	z = X*theta;

	h = sigmoid(z);
	h2 = 1 - h;

	y2 = 1 - y;

	l1 = log(h);
	l2 = log(h2);

	J = (transpose(l1)*(-y) - transpose(l2)*y2)/m;

	h2 = h - y;

	grad = (transpose(X)*h2)/m;









% =============================================================

end
