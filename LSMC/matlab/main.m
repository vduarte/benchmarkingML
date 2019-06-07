Spot = 36;
sigma = 0.2;
K = 40;
r = 0.06;
order = 10

eps = 1e-2
compute_price(order, Spot, sigma, K, r)  % warmup
tic()
P = compute_price(order, Spot, sigma, K, r)
dP_dS = (compute_price(Spot + eps, sigma, K, r) - P) / eps
dP_dsigma = (compute_price(Spot, sigma + eps, K, r) - P) / eps
dP_dK = (compute_price(Spot, sigma, K + eps, r) - P) / eps
dP_dr = (compute_price(Spot, sigma, K, r + eps) - P) / eps
t1 = toc()
disp(t1 * 1000)