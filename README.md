``` 
TODO: PRED UKONCENIM PROJEKTU MUSIM ODEBRAT Z .gitignore CLAUDE CODE SOUBORY, ABY TO BYLO HEZCI
```

# Linear Least Squares Methods


## Abstract

This project implements and compares three different
  computational approaches for linear least squares
  regression: **Pure Python** (using only for-loops),
  **NumPy** (vectorized operations), and **Numba**
  (JIT-compiled Python). Each engine implements four
  regression models: **Linear Regression**, **Ridge Regression**,
  **Lasso Regression**, and **Elastic Net Regression**.

  The project provides **performance benchmarking** to
  demonstrate the computational advantages of different
  implementation strategies, particularly showcasing
  Numba's JIT compilation performance gains. Additionally,
  the program offers **curve fitting capabilities** for **sixteen
   pre-selected functions**, allowing users to fit various
  mathematical models to their datasets through an
  interactive menu system.

## Usage

Run the `main.py` file from the root folder and follow the instructions in the terminal.

When running `manual` mode with more trials, warnings may appear - this is normal and nothing to worry about. It simply means the model is searching for hyperparameter types that don't match, and the model reports this. This is expected behavior and the optimization will complete successfully regardless.

## Sample of program work:
```
===== Precipitation Forecasting using Hidden Markov Models =====

(1) - Discrete Hidden Markov Model
(2) - Gaussian Hidden Markov Model with Mixture Emissions
(3) - Variational Gaussian Hidden Markov Model

Press "1", "2" or "3" for choosing your preferable model: 3

Do you would like to backtest only at 20% of Dataset?
(Recommended options for faster backtesting and running time.)

Press "y" for yes or "n" for no: y

================================================================

Forecasting Daily Precipitation Using Variational Gaussian Hidden Markov Model.

Would you like run the model directly with the best hyperparameters found through Bayesian optimization or set hyperparameters yourself?
Type 'auto' to Run with best predefined parameters find by Bayesian Optimization. Bayesian optimization or 'manual' to set hyperparameters yourself:
auto

Starting Variational Gaussian HMM Model Backtesting...

Optimizing hyperparameters...
Optimization Progress: 100%|███████████████| 1/1 [07:40<00:00, 460.92s/it]

Optimizing threshold...
Optimization Progress: 100%|█████████████| 30/30 [00:00<00:00, 210.75it/s]

===== Results with Optimal Threshold =====
Optimal threshold: 0.0963
Accuracy:       0.6459
Precision:      0.5922
Recall:         0.5214
F1 Score:       0.5545
==========================================
```
**Recommendation:**
Run in faster mode with 20% of the training dataset for quicker program execution, otherwise the program runs quite long time.

**Notes:**
In `auto` mode, the most optimal hyperparameters found through Bayesian Optimization are preconfigured.

## Installation

- Clone the repository using SSH or HTTPS
    - **SSH:** `git@github.com:staifmatej/prg-precipitation-forecast-hmm.git`
    - **HTTPS:** `https://github.com/staifmatej/prg-precipitation-forecast-hmm.git`

- Navigate to the project directory

    - `cd prg-precipitation-forecast-hmm`

- Create virtual environment and install dependencies:

    - `python3 -m venv venv`
    - `source venv/bin/activate`
    - `pip install -r requirements.txt`

## Testing

To run the tests, execute `pytest` directly in the main project directory (**root folder**).

## Codestyle

To check code style compliance, run `pylint . --disable=C0301,C0103` from the main project directory. This will analyze all Python files while ignoring line length (C0301) and naming convention (C0103) warnings.
