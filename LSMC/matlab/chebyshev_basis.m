function [B] = chebyshev_basis(x,k)
B= zeros(length(x), k);
B(:,1)=ones(length(x), 1);
B(:,2)=x;
for n=3:k
    B(:,n) = 2.*x'.*B(:, n-1) - B(:, n-2);
end


    