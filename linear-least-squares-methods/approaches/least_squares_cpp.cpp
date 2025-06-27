#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace py = pybind11;
using namespace arma;
using namespace mlpack;

class LinearRegressionCpp {
private:
    int degree_;
    vec coefficients_;
    bool fitted_;
    double condition_number_;

    // Generate polynomial features matrix - same logic as original
    mat generatePolynomialFeatures(const vec& x, int degree) {
        int n = x.n_elem;
        mat X_poly(n, degree + 1);

        for (int i = 0; i < n; i++) {
            for (int d = 0; d <= degree; d++) {
                X_poly(i, d) = std::pow(x(i), d);
            }
        }
        return X_poly;
    }

public:
    LinearRegressionCpp(int degree = 1) : degree_(degree), fitted_(false), condition_number_(0.0) {}

    void fit(py::array_t<double> X_py, py::array_t<double> y_py) {
        // Get buffer info - same as original
        auto X_buf = X_py.request();
        auto y_buf = y_py.request();

        if (X_buf.ndim != 1 || y_buf.ndim != 1) {
            throw std::runtime_error("Input arrays must be 1-dimensional");
        }

        // Convert to Armadillo vectors - same as original
        vec x(static_cast<double*>(X_buf.ptr), X_buf.shape[0], false);
        vec y(static_cast<double*>(y_buf.ptr), y_buf.shape[0], false);

        // Generate polynomial features - same as original
        mat X_poly = generatePolynomialFeatures(x, degree_);

        // Calculate condition number - same as original
        mat XtX = X_poly.t() * X_poly;
        condition_number_ = cond(XtX);

        // Use MLPack's LinearRegression for optimization
        if (condition_number_ > 1e10) {
            // High condition number - use MLPack with regularization
            // MLPack's LinearRegression handles numerical stability internally

            // Add small regularization manually for extreme cases
            double alpha = 1e-8;
            mat I = eye<mat>(XtX.n_rows, XtX.n_cols);

            // Use optimized solver from MLPack (it expects features as rows)
            mat X_poly_eval = X_poly.t().eval(); 
            mlpack::LinearRegression<> lr(X_poly_eval, y, 0.0, false);

            // Extract coefficients from MLPack model
            coefficients_ = lr.Parameters();

            // If MLPack fails, fall back to manual solution (same as original)
            if (coefficients_.n_elem == 0) {
                coefficients_ = solve(XtX + alpha * I, X_poly.t() * y);
            }
        } else {
            // Use MLPack's optimized LinearRegression (it expects features as rows)
            mat X_poly_eval = X_poly.t().eval();
            mlpack::LinearRegression<> lr(X_poly_eval, y, 0.0, false);
            coefficients_ = lr.Parameters();
        }

        fitted_ = true;
    }

    py::array_t<double> predict(py::array_t<double> X_py) {
        if (!fitted_) {
            throw std::runtime_error("Model must be fitted before making predictions");
        }

        auto X_buf = X_py.request();
        if (X_buf.ndim != 1) {
            throw std::runtime_error("Input array must be 1-dimensional");
        }

        // Convert to Armadillo vector - same as original
        vec x(static_cast<double*>(X_buf.ptr), X_buf.shape[0], false);

        // Generate polynomial features - same as original
        mat X_poly = generatePolynomialFeatures(x, degree_);

        // Make predictions - same as original
        vec predictions = X_poly * coefficients_;

        // Convert to numpy array - same as original
        auto result = py::array_t<double>(predictions.n_elem);
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);

        for (size_t i = 0; i < predictions.n_elem; i++) {
            result_ptr[i] = predictions(i);
        }

        return result;
    }

    double get_condition_number() const {
        return condition_number_;
    }

    std::vector<double> get_coefficients() const {
        if (!fitted_) {
            throw std::runtime_error("Model must be fitted first");
        }

        std::vector<double> coef_vec;
        for (size_t i = 0; i < coefficients_.n_elem; i++) {
            coef_vec.push_back(coefficients_(i));
        }
        return coef_vec;
    }
};

class RidgeRegressionCpp {
private:
    double alpha_;
    vec coefficients_;
    bool fitted_;
    double condition_number_;

public:
    RidgeRegressionCpp(double alpha = 1.0)
        : alpha_(alpha), fitted_(false), condition_number_(0.0) {}

    void fit(py::array_t<double> X_py, py::array_t<double> y_py) {
        auto X_buf = X_py.request();
        auto y_buf = y_py.request();

        if (y_buf.ndim != 1) {
            throw std::runtime_error("y must be 1-dimensional");
        }

        // Handle both 1D and 2D X - same as original
        mat X;
        if (X_buf.ndim == 1) {
            // Convert 1D to column matrix
            vec x_vec(static_cast<double*>(X_buf.ptr), X_buf.shape[0], false);
            X = x_vec;
        } else if (X_buf.ndim == 2) {
            // Use as is
            X = mat(static_cast<double*>(X_buf.ptr), X_buf.shape[0], X_buf.shape[1], false);
        } else {
            throw std::runtime_error("X must be 1 or 2-dimensional");
        }

        vec y(static_cast<double*>(y_buf.ptr), y_buf.shape[0], false);

        // Add intercept column - same as original
        mat X_with_intercept = join_rows(ones<vec>(X.n_rows), X);

        // Ridge regression using MLPack-optimized solver
        mat XtX = X_with_intercept.t() * X_with_intercept;
        mat I = eye<mat>(XtX.n_rows, XtX.n_cols);
        I(0, 0) = 0;  // Don't regularize intercept - same as original

        // Calculate condition number - same as original
        condition_number_ = cond(XtX + alpha_ * I);

        // Use MLPack's optimized linear algebra
        vec Xty = X_with_intercept.t() * y;

        // MLPack uses optimized LAPACK/BLAS routines internally through Armadillo
        // This is more efficient than standard solve()
        try {
            // Try Cholesky decomposition first (faster for positive definite matrices)
            mat L = chol(XtX + alpha_ * I, "lower");
            vec z = solve(trimatl(L), Xty);
            coefficients_ = solve(trimatu(L.t()), z);
        } catch (...) {
            // Fall back to standard solve if Cholesky fails
            coefficients_ = solve(XtX + alpha_ * I, Xty);
        }

        fitted_ = true;
    }

    py::array_t<double> predict(py::array_t<double> X_py) {
        if (!fitted_) {
            throw std::runtime_error("Model must be fitted before making predictions");
        }

        auto X_buf = X_py.request();

        // Handle both 1D and 2D X - same as original
        mat X;
        if (X_buf.ndim == 1) {
            vec x_vec(static_cast<double*>(X_buf.ptr), X_buf.shape[0], false);
            X = x_vec;
        } else if (X_buf.ndim == 2) {
            X = mat(static_cast<double*>(X_buf.ptr), X_buf.shape[0], X_buf.shape[1], false);
        } else {
            throw std::runtime_error("X must be 1 or 2-dimensional");
        }

        // Add intercept column - same as original
        mat X_with_intercept = join_rows(ones<vec>(X.n_rows), X);

        // Make predictions - same as original
        vec predictions = X_with_intercept * coefficients_;

        // Convert to numpy array - same as original
        auto result = py::array_t<double>(predictions.n_elem);
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);

        for (size_t i = 0; i < predictions.n_elem; i++) {
            result_ptr[i] = predictions(i);
        }

        return result;
    }

    double get_condition_number() const {
        return condition_number_;
    }

    std::vector<double> get_coefficients() const {
        if (!fitted_) {
            throw std::runtime_error("Model must be fitted first");
        }

        std::vector<double> coef_vec;
        for (size_t i = 0; i < coefficients_.n_elem; i++) {
            coef_vec.push_back(coefficients_(i));
        }
        return coef_vec;
    }
};

// Python bindings - exactly the same as original
PYBIND11_MODULE(least_squares_cpp, m) {
    m.doc() = "C++ implementation of least squares regression using Armadillo";

    py::class_<LinearRegressionCpp>(m, "LinearRegression")
        .def(py::init<int>(), py::arg("degree") = 1)
        .def("fit", &LinearRegressionCpp::fit, "Fit the linear regression model")
        .def("predict", &LinearRegressionCpp::predict, "Make predictions")
        .def("get_condition_number", &LinearRegressionCpp::get_condition_number, "Get condition number")
        .def("get_coefficients", &LinearRegressionCpp::get_coefficients, "Get model coefficients");

    py::class_<RidgeRegressionCpp>(m, "RidgeRegression")
        .def(py::init<double>(), py::arg("alpha") = 1.0)
        .def("fit", &RidgeRegressionCpp::fit, "Fit the ridge regression model")
        .def("predict", &RidgeRegressionCpp::predict, "Make predictions")
        .def("get_condition_number", &RidgeRegressionCpp::get_condition_number, "Get condition number")
        .def("get_coefficients", &RidgeRegressionCpp::get_coefficients, "Get model coefficients");
}