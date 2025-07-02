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
  regression models from scratch: **Linear Regression** and
   **Ridge Regression** are implemented entirely without
  external libraries, while **Lasso Regression** and
  **Elastic Net Regression** utilize only `from
  sklearn.linear_model import Lasso, ElasticNet` for
  comparison purposes.

  This project extends beyond the scope of the **Linear 
  Algebra II** course at **CTU FIT**, where the least
  squares method was introduced theoretically. As an
  extension of the coursework, I explored practical
  implementation without relying on high-level machine
  learning libraries, demonstrating the mathematical
  foundations learned in class through code.

  The project provides **performance benchmarking** to
  demonstrate the computational advantages of different
  implementation strategies, particularly showcasing
  Numba's JIT compilation performance gains over pure
  Python implementations. Additionally, the program offers
  **curve fitting capabilities** for **sixteen pre-selected
   functions**, allowing users to fit various mathematical
  models to their datasets through an interactive menu
  system.

For a practical demonstration of the program without needing to run it locally, an interactive example is
  available through the Jupyter notebook
  [run_example.ipynb](run_example.ipynb) file included in this repository


For a more detailed description of the methodology, results, and analysis, please refer to the [staifmatej-report.pdf](staifmatej-report.pdf) file included in this repository.


## Usage

Run the `main.py` file from the root folder and follow the instructions in the terminal.

**Recommendation**: For the fastest way to try the program, it is advisable to use the pre-selected "Use example dataset" option.




## Installation

- Clone the repository using SSH or HTTPS
    - **SSH:** `git@github.com:staifmatej/linear-least-squares-methods.git`
    - **HTTPS:** `https://github.com/staifmatej/linear-least-squares-methods.git`

- Navigate to the project directory

    - `cd linear-least-squares-methods`

- Create virtual environment and install dependencies:

    - `python3 -m venv venv`
    - `source venv/bin/activate`
    - `pip install -r requirements.txt`
 
    - `python main.py` (for start program)
    - `pytest` (for testing unit tests)
    - `pylint . --disable=C0301,C0103` (for PEP8 score)

## Testing

To run the tests, execute `pytest` directly in the main project directory (**root folder**).

## Codestyle

To check code style compliance, run `pylint . --disable=C0301,C0103` from the main project directory. This will analyze all Python files while ignoring line length (C0301) and naming convention (C0103) warnings.
