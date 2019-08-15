
order = 25;

Spot = 36;
sigma = 0.2;
K = 40;
r = 0.06;
n = 100000;
m = 10;
T = 1;
delta_t = T / m;

eps = 1e-2;
compute_price(order, Spot, sigma, K, r, n, m, delta_t)  % warmup
tic()
P = compute_price(order, Spot, sigma, K, r, n, m, delta_t);
dP_dS = (compute_price(order, Spot + eps, sigma, K, r, n, m, delta_t) - P) / eps;
dP_dsigma = (compute_price(order, Spot, sigma + eps, K, r, n, m, delta_t) - P) / eps;
dP_dK = (compute_price(order, Spot, sigma, K + eps, r, n, m, delta_t) - P) / eps;
dP_dr = (compute_price(order, Spot, sigma, K, r + eps, n, m, delta_t) - P) / eps;
t1 = toc();
disp(t1 * 1000)


function [output] = chebyshev_basis(x,k)
    B = ones(k, length(x));
    B(2, :)=x;
    for n=3:k
        B(n, :) = 2.*x.*B(n-1, :) - B(n-2, :);
    end
    output = B';
end

function output = ridge_regression(X, Y, lambda)
    I = eye(size(X, 2));
    beta = (X' * X + lambda * I) \ (X' * Y);
    output = X * beta;
end

function [output] = first_one(x)
    original = x;
    x = x > 0;
    nt = length(x(1,:));
    batch_size = length(x(:,1));
    x_not=not(x);
    sum_x=min(cumprod(x_not,2),1.);
    v_ones = ones(batch_size,1);
    lag = sum_x(:, 1:(nt - 1));
    lag = [v_ones lag];
    output= original.*(lag.* x);  
end

function output = scale(x)
        xmin = min(x);
        xmax = max(x);
        a = 2 / (xmax - xmin);
        b = -0.5 * a * (xmin + xmax);
        output = a * x + b;
end

function output = advance(S, r, sigma, delta_t, n)
    dB = sqrt(delta_t)*randn(1,n);
    output = S + r * S * delta_t + sigma * S .* dB;
end

function output = compute_price(order, Spot, sigma, K, r,...
                                n, m, delta_t)
    rng(0);
    S=zeros(m+1,n);
    S(1,:) = Spot;
    for t=2:m+1
        S(t,:) = advance(S(t-1,:), r ,sigma, delta_t, n);
    end

    discount = exp(-r * delta_t);
    CFL=max(0,K-S);
    value = zeros(m,n);
    value(end,:) = CFL(end,:)*discount;
    CV=zeros(m,n);
    for k=1:m-1
        t=m-k;
        t_next = t + 1;
        XX = chebyshev_basis(scale(S(t_next,:)), order);
        YY = value(t_next,:)';
        CV(t,:) = ridge_regression(XX, YY, 100);
        value(t,:) = discount * where(CFL(t_next,:)>CV(t,:),...
                                      CFL(t_next,:),...
                                      value(t_next,:));
    end
    POF = where(CV < CFL(2:end, :), CFL(2:end, :), 0 * CFL(2:end, :))';
    FPOF = first_one(POF);
    dFPOF = (FPOF.*exp(-r*(0:m-1) * delta_t));
    PRICE = mean(sum(dFPOF,2));
    output=PRICE;
end

function out = where(cond, true, false)
    out = true .* (cond) + (1 - cond) .* false;
end