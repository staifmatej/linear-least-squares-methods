"""LeastSquares implementation in Python with numpy."""

import numpy as np
from sklearn.linear_model import coordinate_descent


class LeastSquares:
    """LeastSquares implementation just with Numpy library."""

    def __init__(self, type_regression="PolynomialRegression"):
        self.type_regression=type_regression

        types_of_regression = ["PolynomialRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]
        count = 0
        for i, type_name in enumerate(types_of_regression):
            if type_regression == type:
                count += 1
                break
            if i == len(types_of_regression) and count == 0:
                raise ValueError(f"Type {self.type_regression} is not a valid predefined type.")

    def multivariate_ols(self, X, Y):
        """
            X: feature matrix
                Format: (N, p), where N >= p and N = number of observations, p = number of features
            Y: target variable vector
                Format: (N, ) - one-dimensional vector
            return: w = coefficient vector [w₀, w₁, w₂, ..., wₚ]
        """

        X = np.column_stack([np.ones(len(Y)), X])

        n_rows, n_cols = X.shape
        if n_rows < n_cols:
            raise ValueError(f"Matrix X must have more rows than columns. Got {n_rows} rows and {n_cols} columns")

        treshold_for_QR_decomposition = 1e6
        if self.type_regression == "RidgeRegression":
            """Modified Ridge Regression implementation."""
            XtX = X.T @ X
            regularization = self.alpha * np.eye(X.shape[1])
            regularization[0, 0] = 0  # Neregularizuj intercept
            XtX_modified = XtX + regularization
            cond_number = np.linalg.cond(XtX_modified)
            
            if cond_number < treshold_for_QR_decomposition:
                w = self.normal_equations_ridge(X, Y)
            else:
                w = self.qr_decomposition_ridge(X, Y)

        elif self.type_regression == "PolynomialRegression":
            """ Standard LeastSquares for Standard PolynomialRegression """
            cond_number = np.linalg.cond(X.T @ X)
            if cond_number < treshold_for_QR_decomposition:
                w = self.normal_equations(X, Y)
            else:
                w = self.qr_decomposition(X, Y)

        return w

    def normal_equations(self, X, Y):
        """Compute LeastSquares coefficients using normal equations method with singular matrix protection."""
        XtY =  X.T @ Y
        XtX = X.T @ X
        try:
            w = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XtX) @ XtY
        return w

    def qr_decomposition(self, X, Y):
        """Compute LeastSquares coefficients using QR decomposition method for better numerical stability."""
        Q, R = np.linalg.qr(X)
        QtY = Q.T @ Y
        try:
            w = np.linalg.solve(R, QtY)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(R) @ QtY
        return w
    
    def normal_equations_ridge(self, X, Y):
        """Ridge normal equations - integrovány do LeastSquares."""
        XtX = X.T @ X
        regularization = self.alpha * np.eye(X.shape[1])
        regularization[0, 0] = 0 
        XtX_ridge = XtX + regularization
        
        try:
            return np.linalg.solve(XtX_ridge, X.T @ Y)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(XtX_ridge) @ X.T @ Y
    
    def qr_decomposition_ridge(self, X, Y):
        """Ridge QR dekompozice - integrována do LeastSquares."""
        n_features = X.shape[1]
        sqrt_alpha = np.sqrt(self.alpha)
        
        I_reg = np.eye(n_features)
        I_reg[0, 0] = 0
        X_extended = np.vstack([X, sqrt_alpha * I_reg])
        Y_extended = np.hstack([Y, np.zeros(n_features)])
        
        Q, R = np.linalg.qr(X_extended)
        QtY = Q.T @ Y_extended
        try:
            return np.linalg.solve(R, QtY)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(R) @ QtY

class PolynomialRegression(LeastSquares):
    """Standard Polynomial regression using LeastSquares"""

    def __init__(self, type_regression="PolynomialRegression"):
        super().__init__(type_regression="PolynomialRegression")
        self.degree = degree
        self.coefficients = None

    def fit(self, x, y):
        """Fit polynomial regression model."""
        X_polynomial = self._generate_polynomial_features(x)
        self.coefficients = self.multivariate_ols(X_polynomial, y)
        return self

    def predict(self, x):
        """Predict using fitted polynomial model."""
        X_polynomial = self._generate_polynomial_features(x)
        X_polynomial_with_intercept = np.column_stack([np.ones(len(x)), X_polynomial])
        return X_polynomial_with_intercept @ self.coefficients

    def _generate_polynomial_features(self, x):
        """Generate polynomial features up to specified degree."""
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        polynomial_features = []
        for i in range(1, self.degree + 1):
            polynomial_features.append(x**i)

        X_polynomial = np.column_stack(polynomial_features)

        return X_polynomial

class RidgeRegression(LeastSquares):
    """Ridge regression using LeastSquares infrastructure"""

    def __init__(self, alpha=1.0):
        super().__init__(type_regression="RidgeRegression", alpha=alpha)

    def fit(self, x, y):
        # Připrav data stejně jako LeastSquares
        X_with_intercept = np.column_stack([np.ones(len(y)), x])
        
        # Použij přímo LeastSquares multivariate_ols - detekuje RidgeRegression automaticky
        self.coefficients = self.multivariate_ols(X_with_intercept, y)
        return self
    
    def predict(self, x):
        """Prediction using Ridge coefficients."""
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Stejný approach jako LeastSquares
        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients


class LassoRegression(LeastSquares):
    """Lasso regression using LeastSquares infrastructure + sklearn coordinate descent"""
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha  # L1 regularization strength
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, x, y):
        # Použij LeastSquares preprocessing - přidej intercept
        X_with_intercept = np.column_stack([np.ones(len(y)), x])
        
        # Použij LeastSquares logiku pro data validation a setup
        self.coefficients = self.multivariate_ols_lasso(X_with_intercept, y)
        return self
    
    def multivariate_ols_lasso(self, X, y):
        """Využívá LeastSquares strukturu s Lasso coordinate descent."""
        # Zkontroluj condition number jako LeastSquares
        condition_number = np.linalg.cond(X.T @ X)
        print(f"Lasso: Matrix condition number: {condition_number:.2e}")
        
        if condition_number > 1e12:
            print("Lasso: Ill-conditioned matrix detected, regularization will help")
        
        # Použij coordinate descent pro Lasso
        return self._coordinate_descent_lasso(X, y)
    
    def _coordinate_descent_lasso(self, X, y):
        """Lasso coordinate descent - používá sklearn implementaci."""

        # Oddělej intercept a features (LeastSquares approach)
        X_features = X[:, 1:]  # Bez intercept sloupce
        
        # Centruj data (standardní LeastSquares preprocessing)
        X_centered = X_features - np.mean(X_features, axis=0)
        y_centered = y - np.mean(y)
        
        # Použij sklearn coordinate descent
        coefficients_features, dual_gap, _, n_iter = coordinate_descent.lasso_path(
            X_centered, y_centered, 
            alphas=[self.alpha],
            max_iter=self.max_iter,
            tol=self.tol,
            return_n_iter=True
        )
        
        print(f"Lasso: Converged in {n_iter[0]} iterations, dual gap: {dual_gap[0]:.2e}")
        
        # Rekonstruuj intercept (LeastSquares způsob)
        feature_coeffs = coefficients_features[:, 0]
        intercept = np.mean(y) - np.mean(X_features, axis=0) @ feature_coeffs
        
        # Vrať v LeastSquares formatu [intercept, features]
        return np.concatenate([[intercept], feature_coeffs])
    
    def predict(self, x):
        """Použij stejnou prediction logiku jako LeastSquares."""
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Stejný approach jako LeastSquares
        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients


class ElasticNetRegression(LeastSquares):
    """Elastic Net regression using LeastSquares infrastructure + Ridge + Lasso techniques"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha  # Overall regularization strength
        self.l1_ratio = l1_ratio  # L1 vs L2 mix: 0=Ridge, 1=Lasso, 0.5=Equal mix
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, x, y):
        # Použij LeastSquares preprocessing - stejně jako Ridge/Lasso
        X_with_intercept = np.column_stack([np.ones(len(y)), x])
        
        # Použij kombinovanou logiku Ridge + Lasso
        self.coefficients = self.multivariate_ols_elasticnet(X_with_intercept, y)
        return self
    
    def multivariate_ols_elasticnet(self, X, y):
        """Využívá LeastSquares + Ridge + Lasso infrastrukturu pro ElasticNet."""
        # Zkontroluj condition number jako Ridge/Lasso
        condition_number = np.linalg.cond(X.T @ X)
        print(f"ElasticNet: Matrix condition number: {condition_number:.2e}")
        
        # Vypočti L1 a L2 alpha hodnoty
        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1 - self.l1_ratio)
        
        print(f"ElasticNet: L1 penalty (α₁): {alpha_l1:.4f}, L2 penalty (α₂): {alpha_l2:.4f}")
        
        if condition_number > 1e12:
            print("ElasticNet: Ill-conditioned matrix detected, regularization will help")
        
        # Použij coordinate descent pro ElasticNet (kombinuje L1+L2)
        return self._coordinate_descent_elasticnet(X, y, alpha_l1, alpha_l2)
    
    def _coordinate_descent_elasticnet(self, X, y, alpha_l1, alpha_l2):
        """ElasticNet coordinate descent - kombinuje Ridge a Lasso techniky."""
        # Oddělej intercept a features (stejně jako Lasso)
        X_features = X[:, 1:]  # Bez intercept sloupce
        
        # Centruj data (LeastSquares preprocessing)
        X_centered = X_features - np.mean(X_features, axis=0)
        y_centered = y - np.mean(y)
        
        # Použij sklearn elastic net coordinate descent
        coefficients_features, dual_gap, _, n_iter = coordinate_descent.enet_path(
            X_centered, y_centered,
            l1_ratio=self.l1_ratio,
            alphas=[self.alpha],
            max_iter=self.max_iter,
            tol=self.tol,
            return_n_iter=True
        )
        
        print(f"ElasticNet: Converged in {n_iter[0]} iterations, dual gap: {dual_gap[0]:.2e}")
        
        # Rekonstruuj intercept (LeastSquares způsob - stejně jako Lasso)
        feature_coeffs = coefficients_features[:, 0]
        intercept = np.mean(y) - np.mean(X_features, axis=0) @ feature_coeffs
        
        # Vrať v LeastSquares formatu [intercept, features]
        return np.concatenate([[intercept], feature_coeffs])
    
    def predict(self, x):
        """Použij stejnou prediction logiku jako LeastSquares/Ridge/Lasso."""
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Identický approach jako LeastSquares
        X_with_intercept = np.column_stack([np.ones(len(x)), x])
        return X_with_intercept @ self.coefficients
