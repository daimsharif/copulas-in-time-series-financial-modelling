# Copula-Based Financial Risk Analysis

This project simulates, analyzes, and visualizes multivariate dependencies and systemic risk in financial time series using **copula models** — both general and time-series-based. It is built using synthetic data to model realistic behaviors like volatility clustering, tail dependence, and stress scenarios. The summary report of our findings is in the 

---

## 📁 Directory Structure

```
src/
├── main.py                  # Main entry point, runs full copula analysis pipeline
├── copula/
│   ├── CopulaModel.py
│   ├── Marginal.py
│   ├── DataAnalyzer.py
│   ├── StudentTCopula.py
│   ├── ClaytonCopula.py
│   └── TimeSeries/
│       ├── CoVaRCopula.py
│       ├── DCCCopula.py
│       └── GARCHVineCopula.py
```

---

## 🔍 What It Does

### Part 1: General Copula Analysis

* Creates a synthetic dataset with non-Gaussian marginals.
* Introduces dependencies using a **Student-t copula**.
* Measures:

  * Pearson & Spearman correlations
  * Tail dependence
* Visualizes:

  * Scatter matrix
  * Joint KDE plot
  * Bivariate **Clayton copula** sample

### Part 2: Time-Series Financial Risk Modeling

* Simulates daily returns for 4 financial assets with realistic volatility dynamics.
* Inserts a **stress period** with higher correlations and volatilities.
* Models risk using:

  * **GARCH-Vine Copula**
  * **DCC Copula**
  * **CoVaR Copula**
* Computes:

  * Value-at-Risk (VaR)
  * Conditional VaR (CVaR)
  * CoVaR
  * Systemic Impact

---

## 📊 Output Files

* `initial_timeseries_per_asset.png`
* `scatter_matrix.png`
* `joint_distribution.png`
* `clayton_samples.png`
* `financial_returns.png`

---

## 🚀 How to Run

```bash
# From project root
cd src
python main.py
```

---

## 🧰 Requirements

Install dependencies (assuming `requirements.txt` exists):

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib seaborn scipy
```

> Ensure all local copula module dependencies (like `CopulaModel`, `DCCCopula`, etc.) are present in `copula/` folder as shown above.

---

## 📌 Notes

* All visualizations are saved automatically.
* Random seed is set for reproducibility.
* Works entirely on **synthetic data** — no financial data is required.

---

## 📧 Author
**Aryan**

