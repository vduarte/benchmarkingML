// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include <math.h>
#include <chrono>

using namespace Rcpp;
using namespace arma;

const double d_gamma = 2.0;
const double d_one_minus_gamma = 1-d_gamma;

const auto uLambda = [](double val) { return ::pow(val, d_one_minus_gamma) / d_one_minus_gamma; };

const auto clampLambda = [](double val) { return val < 1e-14 ? 1e-14 : val; };

// [[Rcpp::export]]
List dmain(const colvec& logy_grid, const mat& Py, int nB, int repeats) {
    double beta = 0.953;
    double r = 0.017;
    double theta = 0.282;
    double theta_comp = 1 - theta;
    unsigned int ny = logy_grid.n_rows;

    colvec Bgrid = linspace<colvec>(-0.45, 0.45, nB);
    colvec ygrid = exp(logy_grid);

    double ymean = mean(ygrid);

    colvec u_def_y(ny);
    u_def_y.fill(ymean * 0.969);
    u_def_y = arma::min(u_def_y, ygrid);
    u_def_y.transform(uLambda);

    colvec Vd = zeros<colvec>(ny);
    mat Vc = zeros<mat>(ny, nB);
    mat V = zeros<mat>(ny, nB);

    mat Q(ny, nB);
    Q.fill(0.95);

    mat myY(ygrid.begin(), ny, 1);
    mat myBnext(Bgrid.begin(), 1, nB);

    int zero_ind = nB / 2;

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < repeats; ++i) {
        mat betaEV = beta * Py * V;
        mat EVd = Py * Vd;
        mat EVc_zero_ind = Py * Vc.col(zero_ind);
        mat Vd_target = u_def_y + beta * (theta * EVc_zero_ind + theta_comp * EVd);

        mat sliceTemplate = -Q;
        sliceTemplate.each_row() %= myBnext;
        sliceTemplate.each_col() += myY;

        mat Vc_target(ny, nB);

        for (unsigned int j = 0; j < nB; ++j) {
            mat oneSlice = sliceTemplate;
            oneSlice += Bgrid(j);
            oneSlice.transform(clampLambda);
            oneSlice.transform(uLambda);
            oneSlice += betaEV;
            Vc_target.col(j) = arma::max(oneSlice, 1);
        }

        mat Vdd = repmat(Vd, 1, nB);

        umat default_states = Vdd > Vc;
        mat default_prob = Py * default_states;

        Q = (1 - default_prob) / (1 + r);
        V = arma::max(Vc, Vdd);
        Vc = Vc_target;
        Vd = Vd_target;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto out = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()*1.0/repeats;

    return List::create(Named("V") = V,
            Named("millis") = out
            );
}
