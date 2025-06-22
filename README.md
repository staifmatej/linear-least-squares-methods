# Linear Least Squares Methods

This project explores the implementation of the **Least Squares Method** for curve fitting using four different approaches:

1. Python using NumPy  
2. Python using raw `for` loops  
3. Python with `for` loops accelerated using Numba  
4. C++ implementation  

## Project Goals

- Implement the least squares method **from scratch**, without relying on high-level libraries (except for the NumPy version).
- Build an interface that allows users to input their own data points and select the desired fitting function (e.g. linear, quadratic, etc.).
- Visualize the resulting fits using **Python plotting** (e.g. Matplotlib), regardless of which backend implementation was used.
- Compare performance between the different implementations.
- Enable **polynomial regression up to degree 5**.
- Extend the project to support alternative regression methods:
  - **Standard Regression**  
  - **Lasso Regression**
  - **Ridge Regression**
  - **Elastic Net Regression**

## Future Plans

- CLI or web interface for user input
- Performance benchmarks
- Integration of scikit-learn for advanced regression models
