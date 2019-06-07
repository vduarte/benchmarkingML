function [output] = ridge_regression(X, Y, lambda)
    I = eye(size(X, 2));
    beta = (X' * X + lambda * I) \ (X' * Y');
    output = X * beta;
end

