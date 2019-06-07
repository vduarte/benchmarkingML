#include <armadillo>

#include <math.h>
#include <chrono>

#include <iostream>

using namespace arma;

const double S0 = 36;
const double sigma = 0.2;
const double K = 40;
const double r = 0.06;
const unsigned int n = 100000;
const unsigned int m = 10;
const int T = 1;
const double delta_t = (T*1.0)/m;

rowvec advance(const rowvec& S) {
    rowvec dB = sqrt(delta_t) * randn<rowvec>(n);
    rowvec out = S % ((1 + r * delta_t) + sigma * dB);
    return out;
}

rowvec scale(const rowvec& x) {
    double xmin = arma::min(x);
    double xmax = arma::max(x);
    double a = 2 / (xmax - xmin);
    double b = 1 - a * xmax;
    return a * x + b;
}
        
mat chebyshev_basis(const rowvec& x, int k) {
    mat B(x.n_elem, k);
    B.col(0).ones();
    colvec xt = x.t();
    B.col(1) = xt;
    xt *= 2;
    for(int n = 2; n < k; n++) {
        B.col(n) = xt;
        B.col(n) %= B.col(n - 1);
        B.col(n) -= B.col(n - 2);
    }
    return B;
}

rowvec ridge_regression(const mat& X, const rowvec& Y, int order, double lambda = 100) {
    mat I = mat(order, order, fill::eye);
    vec betta = arma::solve(X.t() * X + lambda * I, X.t() * Y.t());
    return (X * betta).t();
}

mat first_one(const mat& original) {
    mat x = arma::conv_to<mat>::from(original > 0);
    unsigned int nt = x.n_rows;
    unsigned int batch_size = x.n_cols;
    mat x_not = 1.0 - x;
    mat sum_x = arma::min(arma::cumprod(x_not, 0), ones<mat>(nt, batch_size));

    mat lag(nt, batch_size);
    lag.row(0) = ones<rowvec>(batch_size);
    lag.rows(1, nt-1) = sum_x.rows(0, nt-2);

    return original % (lag % x);
}

double main_fun(unsigned int order) {
    mat S(m + 1, n); // m + 1 x n
    S.row(0).fill(S0);
    
    for(unsigned int t = 0; t < m; t++) {
        S.row(t + 1) = advance(S.row(t));
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
        
        mat X = chebyshev_basis(scale(S.row(t_next)), order);
        CV.row(t) = ridge_regression(X, value.row(t_next), order);

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

    double out =  arma::accu(dFPOF) / n;
    return out;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "You need to supply parameter 'order', for example:" << std::endl
                  << argv[0] << " 10" << std::endl;
        return 1;
    }

    int order = atoi(argv[1]);

    if (order <= 0) {
        std::cout << "Order must be a positive number." << std::endl
                  << "Received: " << argv[1] << std::endl
                  << "Parsed: " << order << std::endl
                  << "Aborting!" << std::endl;
        return 2;
    }

    std::cout << "order: " << order << std::endl;

//    arma_rng::set_seed(0);
    arma_rng::set_seed_random();
    auto t0 = std::chrono::steady_clock::now();
    double result = main_fun(order);
    auto t1 = std::chrono::steady_clock::now();
    auto out_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    cout.precision(17);
    std::cout << "Out: " << result << std::endl;
    std::cout << "Time (millis): " << out_time * 5 << std::endl;
    return 0;
}
