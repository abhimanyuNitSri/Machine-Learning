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

	
	a = ones([m; 1]);

	x1 = [a, X];

	z2 = x1*transpose(Theta1); % 5000 x 25

	x2 = z2;
	x2 = sigmoid(z2);

	x2 = [a, x2]; 

	z3 = x2*transpose(Theta2); %
	h = z3;
	h = sigmoid(z3);

	k = num_labels;

	p = zeros(m, 1);

	for i = 1:m
		temp = -1;
		tempj = -1;
		for j = 1:k
			if(h(i, j) > temp)
				temp = h(i,j);
				tempj = j;
			endif
		end

		p(i) = tempj;
	end







% =========================================================================


end
