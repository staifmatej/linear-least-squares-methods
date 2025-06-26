#include <mlpack.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <armadillo>

namespace py = pybind11;
using namespace mlpack;
using namespace mlpack::regression;
using namespace arma;

class LinearRegressionCpp {
private:
    int degree_;
    LinearRegression mlpack_model_;  // Direct MLPack model
    bool fitted_;
    double condition_number_;
    
public:
    LinearRegressionCpp(int degree = 1) : degree_(degree), fitted_(false), condition_number_(0.0) {}
    
    void fit(const py::array_t<double>& X_py, const py::array_t<double>& y_py) {
        auto X_buf = X_py.request();
        auto y_buf = y_py.request();
        
        if (X_buf.ndim != 1 || y_buf.ndim != 1) {
            throw std::runtime_error("Input arrays must be 1-dimensional");
        }
        
        int n_samples = X_buf.shape[0];
        double* X_ptr = static_cast<double*>(X_buf.ptr);
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        
        // Create polynomial features matrix
        mat X_features = generatePolynomialFeatures(X_ptr, n_samples, degree_);
        rowvec y_vec(y_ptr, n_samples);
        
        // Calculate condition number
        condition_number_ = cond(X_features);
        
        // Use MLPack's LinearRegression directly - fastest method!
        mlpack_model_ = LinearRegression(X_features, y_vec, 0.0);  // lambda=0 for OLS
        
        fitted_ = true;
    }
    
    py::array_t<double> predict(const py::array_t<double>& X_py) {
        if (!fitted_) {
            throw std::runtime_error("Model not fitted yet. Call fit() first.");
        }
        
        auto X_buf = X_py.request();
        if (X_buf.ndim != 1) {
            throw std::runtime_error("Input array must be 1-dimensional");
        }
        
        int n_samples = X_buf.shape[0];
        double* X_ptr = static_cast<double*>(X_buf.ptr);
        
        // Generate polynomial features
        mat X_features = generatePolynomialFeatures(X_ptr, n_samples, degree_);
        
        // Use MLPack's direct prediction - fastest!
        rowvec predictions;
        mlpack_model_.Predict(X_features, predictions);
        
        // Convert back to Python array
        auto result = py::array_t<double>(n_samples);
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        for (int i = 0; i < n_samples; ++i) {
            result_ptr[i] = predictions(i);
        }
        
        return result;
    }
    
    py::array_t<double> getCoefficients() {
        if (!fitted_) {
            throw std::runtime_error("Model not fitted yet. Call fit() first.");
        }
        
        // Get coefficients directly from MLPack model
        const vec& coeffs = mlpack_model_.Parameters();
        
        auto result = py::array_t<double>(coeffs.n_elem);
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        for (size_t i = 0; i < coeffs.n_elem; ++i) {
            result_ptr[i] = coeffs(i);
        }
        
        return result;
    }
    
    double getConditionNumber() const {
        return condition_number_;
    }
    
    int getDegree() const {
        return degree_;
    }

private:
    mat generatePolynomialFeatures(double* X, int n_samples, int degree) {
        // Normalize to [0,1] range for numerical stability
        vec X_vec(X, n_samples);
        double x_min = X_vec.min();
        double x_max = X_vec.max();
        
        vec X_normalized = X_vec;
        if (x_max - x_min > 1e-10) {
            X_normalized = (X_vec - x_min) / (x_max - x_min);
        }
        
        // Create polynomial features matrix
        mat X_poly(n_samples, degree);
        for (int d = 1; d <= degree; ++d) {
            X_poly.col(d-1) = pow(X_normalized, d);
        }
        
        return X_poly;
    }
};

class RidgeRegressionCpp {
private:
    double alpha_;
    LinearRegression mlpack_model_;  // Direct MLPack with Ridge regularization
    bool fitted_;
    double condition_number_;
    
public:
    RidgeRegressionCpp(double alpha = 1.0) : alpha_(alpha), fitted_(false), condition_number_(0.0) {}
    
    void fit(const py::array_t<double>& X_py, const py::array_t<double>& y_py) {
        auto X_buf = X_py.request();
        auto y_buf = y_py.request();
        
        int n_samples = X_buf.shape[0];
        int n_features = (X_buf.ndim == 2) ? X_buf.shape[1] : 1;
        
        mat X_mat;
        if (X_buf.ndim == 1) {
            double* X_ptr = static_cast<double*>(X_buf.ptr);
            X_mat = mat(X_ptr, n_samples, 1, false);
        } else {
            double* X_ptr = static_cast<double*>(X_buf.ptr);
            X_mat = mat(X_ptr, n_samples, n_features, false);
        }
        
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        rowvec y_vec(y_ptr, n_samples);
        
        // Calculate condition number
        condition_number_ = cond(X_mat);
        
        // Use MLPack's LinearRegression with Ridge regularization directly!
        mlpack_model_ = LinearRegression(X_mat.t(), y_vec, alpha_);  // lambda=alpha for Ridge
        
        fitted_ = true;
    }
    
    py::array_t<double> predict(const py::array_t<double>& X_py) {
        if (!fitted_) {
            throw std::runtime_error("Model not fitted yet. Call fit() first.");
        }
        
        auto X_buf = X_py.request();
        int n_samples = X_buf.shape[0];
        int n_features = (X_buf.ndim == 2) ? X_buf.shape[1] : 1;
        
        mat X_mat;
        if (X_buf.ndim == 1) {
            double* X_ptr = static_cast<double*>(X_buf.ptr);
            X_mat = mat(X_ptr, n_samples, 1, false);
        } else {
            double* X_ptr = static_cast<double*>(X_buf.ptr);
            X_mat = mat(X_ptr, n_samples, n_features, false);
        }
        
        // Use MLPack's direct prediction - fastest!
        rowvec predictions;
        mlpack_model_.Predict(X_mat.t(), predictions);
        
        auto result = py::array_t<double>(n_samples);
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        for (int i = 0; i < n_samples; ++i) {
            result_ptr[i] = predictions(i);
        }
        
        return result;
    }
    
    py::array_t<double> getCoefficients() {
        if (!fitted_) {
            throw std::runtime_error("Model not fitted yet. Call fit() first.");
        }
        
        // Get coefficients directly from MLPack model
        const vec& coeffs = mlpack_model_.Parameters();
        
        auto result = py::array_t<double>(coeffs.n_elem);
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        for (size_t i = 0; i < coeffs.n_elem; ++i) {
            result_ptr[i] = coeffs(i);
        }
        
        return result;
    }
    
    double getConditionNumber() const {
        return condition_number_;
    }
};

// Python bindings
PYBIND11_MODULE(least_squares_cpp, m) {
    m.doc() = "C++ implementation of least squares regression using MLPack";
    
    py::class_<LinearRegressionCpp>(m, "LinearRegression")
        .def(py::init<int>(), py::arg("degree") = 1)
        .def("fit", &LinearRegressionCpp::fit, "Fit the linear regression model")
        .def("predict", &LinearRegressionCpp::predict, "Make predictions")
        .def("get_coefficients", &LinearRegressionCpp::getCoefficients, "Get model coefficients")
        .def("get_condition_number", &LinearRegressionCpp::getConditionNumber, "Get condition number")
        .def("get_degree", &LinearRegressionCpp::getDegree, "Get polynomial degree")
        .def_property_readonly("coefficients", &LinearRegressionCpp::getCoefficients)
        .def_property_readonly("condition_number", &LinearRegressionCpp::getConditionNumber);
    
    py::class_<RidgeRegressionCpp>(m, "RidgeRegression")
        .def(py::init<double>(), py::arg("alpha") = 1.0)
        .def("fit", &RidgeRegressionCpp::fit, "Fit the Ridge regression model")
        .def("predict", &RidgeRegressionCpp::predict, "Make predictions")
        .def("get_coefficients", &RidgeRegressionCpp::getCoefficients, "Get model coefficients")
        .def("get_condition_number", &RidgeRegressionCpp::getConditionNumber, "Get condition number")
        .def_property_readonly("coefficients", &RidgeRegressionCpp::getCoefficients)
        .def_property_readonly("condition_number", &RidgeRegressionCpp::getConditionNumber);
}