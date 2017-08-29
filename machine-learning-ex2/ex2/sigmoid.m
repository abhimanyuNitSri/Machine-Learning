function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
	g = z;
	row = length(z(:, 1));
	col = length(z(1, :));

	for i = 1:row
		for j = 1:col
			x = z(i,j);
			x = e^(-x);
			x = x+1;
			g(i, j) = 1/x;
		end
	end





% =============================================================

end
