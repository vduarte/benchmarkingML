#include <armadillo>
#include <math.h>
#include <chrono>
#include <iostream>

using namespace arma;


mat chebyshev_basis(const rowvec& x, int k) {
    mat B(x.n_elem, k);
    B.col(0).ones();
    B.col(1) = x.t();
    for(int n = 2; n < k; n++) {
        B.col(n) = 2 * x.t() % B.col(n - 1) - B.col(n - 2);
    }
    return B;
}

rowvec ridge_regression(const mat& X, const rowvec& Y, int order, double lambda = 100) {
    mat I = mat(order, order, fill::eye);
    vec beta = arma::solve(X.t() * X + lambda * I, X.t() * Y.t());
    return (X * beta).t();
}

mat first_one(const mat& original) {
    mat x = arma::conv_to<mat>::from(original > 0);
    unsigned int nt = x.n_rows;
    unsigned int batch_size = x.n_cols;
    mat x_not = 1.0 - x;
    mat sum_x = arma::min(arma::cumprod(x_not, 0), ones<mat>(nt, batch_size));

    mat lag(nt, batch_size);
    mat v_ones = ones<rowvec>(batch_size);
    lag.row(0) = v_ones;
    lag.rows(1, nt-1) = sum_x.rows(0, nt-2);
    mat output = original % (lag % x);
    return output;
}

rowvec scale(const rowvec& x) {
    double xmin = arma::min(x);
    double xmax = arma::max(x);
    double a = 2 / (xmax - xmin);
    double b = -0.5 * a * (xmin + xmax);
    return a * x + b;
}

rowvec advance(const rowvec& S, double r, double sigma, double delta_t, int n) {
    rowvec dB = sqrt(delta_t) * randn<rowvec>(n);
    rowvec output = S + r * S * delta_t + sigma * S % dB;
    return output;
}

rowvec where(rowvec& COND, rowvec& TRUE, rowvec&FALSE) {
    return TRUE % (COND) + (1 - COND) % FALSE;
}


double compute_price(unsigned int order, double Spot, double sigma, double K, double r,
					 int n, int m, double delta_t) {
    arma_rng::set_seed(0);
    mat S(m + 1, n);
    S.row(0).fill(Spot);
    
    for(unsigned int t = 0; t < m; t++) {
        S.row(t + 1) = advance(S.row(t), r, sigma, delta_t, n);
    }
        
    double discount = exp(-r * delta_t);
    mat CFL = arma::max((K - S), zeros<mat>(m + 1, n));
    
    mat value = zeros<mat>(m, n);
    value.row(value.n_rows - 1) = CFL.row(CFL.n_rows - 1) * discount;
    mat CV = zeros<mat>(m, n);
    
    rowvec tmpValue;
    rowvec tmpCFL;

    for(unsigned int k = 1; k < m; k++) {
        unsigned int t = m - k - 1;
        unsigned int t_next = t + 1;
        
        mat XX = chebyshev_basis(scale(S.row(t_next)), order);
        mat YY = value.row(t_next);
        CV.row(t) = ridge_regression(XX, YY, order);

        tmpValue = value.row(t_next);
        tmpCFL = CFL.row(t_next);
        uvec comp = find(tmpCFL > CV.row(t));
        tmpValue.elem(comp) = tmpCFL.elem(comp);
        tmpValue *= discount;
        value.row(t) = tmpValue;
    }

    mat POF = CFL.rows(1, m);
    POF.elem(find(CV >= POF)).zeros();
    mat FPOF = first_one(POF);
    colvec m_range = linspace<colvec>(0, m-1, m);
    mat exp_mat = repmat(arma::exp(-r * m_range * delta_t), 1, n);
    mat dFPOF = FPOF % exp_mat;

    double price =  arma::accu(dFPOF) / n;
    return price;
}


int main(int argc, char** argv) {
    int order = atoi(argv[1]);

    const double Spot = 36;
	const double sigma = 0.2;
	const double K = 40;
	const double r = 0.06;
	const unsigned int n = 100000;
	const unsigned int m = 10;
	const int T = 1;
	const double delta_t = (T*1.0)/m;
	const double epsilon = 1e-2;

    auto t0 = std::chrono::steady_clock::now();
    double P = compute_price(order, Spot, sigma, K, r, n, m, delta_t);
	double dP_dS = (compute_price(order, Spot + epsilon, sigma, K, r, n, m, delta_t) - P) / epsilon;
	double dP_dsigma = (compute_price(order, Spot, sigma + epsilon, K, r, n, m, delta_t) - P) / epsilon;
	double dP_dK = (compute_price(order, Spot, sigma, K + epsilon, r, n, m, delta_t) - P) / epsilon;
	double dP_dr = (compute_price(order, Spot, sigma, K, r + epsilon, n, m, delta_t) - P) / epsilon;
    auto t1 = std::chrono::steady_clock::now();
    auto out_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    cout.precision(5);
    std::cout << "Out: " << P << std::endl;
    std::cout << "Time (millis): " << out_time << std::endl;
    return 0;
}
