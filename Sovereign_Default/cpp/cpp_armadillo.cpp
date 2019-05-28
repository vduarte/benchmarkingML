// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include <math.h>
#include <chrono>

using namespace Rcpp;
using namespace arma;

const double d_gamma = 2.0;
const double d_one_minus_gamma = 1-d_gamma;

// same as function u in python, when applied to each element of the matrix/cube
const auto uLambda = [](double val) { return ::pow(val, d_one_minus_gamma) / d_one_minus_gamma; };

const auto clampLambda = [](double val) { return val < 1e-14 ? 1e-14 : val; };

// [[Rcpp::export]]
List dmain(const colvec& logy_grid, const mat& Py, int nB, int repeats) {
    double beta = 0.953;
    double r = 0.017;
    double theta = 0.282;
    double theta_comp = 1 - theta;
    unsigned int ny = logy_grid.n_rows;

    colvec Bgrid = linspace<colvec>(-0.45, 0.45, nB); // nB
    colvec ygrid = exp(logy_grid); // nB

    double ymean = mean(ygrid);

    colvec u_def_y(ny); // ny
    u_def_y.fill(ymean * 0.969);
    u_def_y = arma::min(u_def_y, ygrid);
    u_def_y.transform(uLambda);
    // def_y was only used as u(def_y), so we can precalculate it

    colvec Vd = zeros<colvec>(ny); // ny x 1
    mat Vc = zeros<mat>(ny, nB); // ny x nB
    mat V = zeros<mat>(ny, nB); // ny x nB

    mat Q(ny, nB); // ny x nB
    Q.fill(0.95);

    mat myY(ygrid.begin(), ny, 1); // ny x 1 instead of ny x 1 x 1
    mat myBnext(Bgrid.begin(), 1, nB); // 1 x nB instead of 1 x 1 x nB

    int zero_ind = nB / 2;

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < repeats; ++i) {
        mat betaEV = beta * Py * V; // ny x nB
        mat EVd = Py * Vd; // ny x 1
        // equivalent: EVc = Py @ Vc; EVc_zero_ind = EVc[:, zero_ind];
        mat EVc_zero_ind = Py * Vc.col(zero_ind); // ny x 1
        // equivalent: EVd == EVd[:, 0], since this matrix has the same dimensions: ny x 1
        mat Vd_target = u_def_y + beta * (theta * EVc_zero_ind + theta_comp * EVd); // ny x 1

        // in python code, the Vc_target was calculated using a three-dimensional matrix.
        // here we use two-dimensional matrices for convenience and memory saving
        // the following is equivalent to the expression: y - Qnext * Bnext
        mat sliceTemplate = -Q; // ny x nB instead of ny x 1 x nB
        sliceTemplate.each_row() %= myBnext; // each_row is 1 x nB, myBnext is 1 x nB,
        // result is still ny x nB instead of ny x 1 x nB
        sliceTemplate.each_col() += myY; // each_col is ny x 1, myY is ny x 1
        // result is still ny x nB instead of ny x 1 x nB

        mat Vc_target(ny, nB); // ny x nB
       	// instead of calculating Vc_target from three-dimensional cube at one,
        // we calculate it slice-by-slice (or column-by-column for Vc_target)
        // each oneSlice corresponds to one slice of cube m in python
        for (unsigned int j = 0; j < nB; ++j) {
            mat oneSlice = sliceTemplate; // ny x nB instead of ny x 1 x nB

            // add Bgrid(j) to each element of the matrix
            oneSlice += Bgrid(j);
            // changes the matrix so that there are no elements less than a certain value
            // equivalent: np.maximum(maxrix, 1e-14)
            oneSlice.transform(clampLambda);
            // equivalent: u(maxrix)
            oneSlice.transform(uLambda);
            // matrix addition, equivalent: u(c) + β * EV
            oneSlice += betaEV;
            Vc_target.col(j) = arma::max(oneSlice, 1);
        }

        mat Vdd = repmat(Vd, 1, nB); // ny x nB

        umat default_states = Vdd > Vc; // ny x nB
        mat default_prob = Py * default_states; // ny x nB

        // equivalent: Q_target = value; Q = Q_target; because variables Q and Q_target are not changed or used between these two lines
        Q = (1 - default_prob) / (1 + r); // ny x nB
        // equivalent: V_upd = value; V = V_upd; because variables V and V_upd are not changed or used between these two lines
        V = arma::max(Vc, Vdd); // ny x nB
        Vc = Vc_target;
        Vd = Vd_target;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto out = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()*1.0/repeats;

    return List::create(Named("V") = V,
            Named("millis") = out
            );
}
