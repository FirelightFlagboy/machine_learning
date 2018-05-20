function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
tab = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
st = size(tab, 2)
error_pre = zeros(st, st);

for ic = 1:st
	tmp_C = tab(ic);
	for is = 1:st
		tmp_s = tab(is);
		fprintf("Training on [%d, %d](%d, %d)\n", ic, is, tmp_C, tmp_s);
		tmp_model = svmTrain(X, y, tmp_C, @(x1, x2) gaussianKernel(x1, x2, tmp_s));
		prediction = svmPredict(tmp_model, Xval);
		error_pre(ic, is) = mean( double(prediction ~= yval) );
	end
end

fprintf("error tab:\n");
disp(error_pre);
fprintf("min error tab:\n");
minval = min( min(error_pre, [], 2));
[r, c] = find(error_pre == minval);
fprintf("at r:%d, c:%d value of :%d\n", r, c, error_pre(r, c));
C = tab(r);
sigma = tab(c);
% =========================================================================

end
