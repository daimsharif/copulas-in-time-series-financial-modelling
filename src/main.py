# main.py – updated 2025‑05‑08
"""
Run the three risk engines **and** output a full copula‑comparison table.

Usage
-----
$ python main.py                           # reads ./data/, α = 0.05
$ DATA_DIR=/path ALPHA=0.01 python main.py  # custom data path
$ USE_SYNTHETIC=1 python main.py           # use synthetic data instead

For synthetic data options:
$ USE_SYNTHETIC=1 N_ASSETS=5 N_OBS=1000 DIST=t DF=6 python main.py
"""

from __future__ import annotations
import os
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import traceback

# Import both real and synthetic data options
from DataLoader import load_returns_from_data_folder
# Import the synthetic data generator
try:
    from synthetic_data_generator import (
        generate_synthetic_returns,
        add_garch_effects,
        generate_example_datasets
    )
except ImportError:
    print("WARNING: synthetic_data_generator module not found in path")
    # Define a placeholder to prevent crashes

    def generate_synthetic_returns(*args, **kwargs):
        print("Synthetic data generation not available.")
        # Return a simple random DataFrame with default parameters
        n_assets = kwargs.get('n_assets', 3)
        n_obs = kwargs.get('n_obs', 500)
        asset_names = kwargs.get(
            'asset_names', [f"Asset_{i+1}" for i in range(n_assets)])
        return pd.DataFrame(
            np.random.randn(n_obs, n_assets) * 0.01,
            columns=asset_names,
            index=pd.date_range(start='2024-01-01', periods=n_obs, freq='B')
        )

    def add_garch_effects(returns, **kwargs):
        print("GARCH effects not available.")
        return returns

    def generate_example_datasets():
        print("Example datasets not available.")
        return {"basic": generate_synthetic_returns()}

# still used for quick diagnostics
from DataAnalyzer import DataAnalyzer

# Attempt to import the copula comparison module
# We'll add error handling to safely handle API changes
try:
    from copula.CopulaComparison import compare_copulas
    # Import the fallback implementation
    # This will be used if the main implementation fails
    try:
        from copula.CopulaComparison import compare_copulas_fallback
    except ImportError:
        # Define a simple fallback function if module doesn't exist
        def compare_copulas_fallback(df, **kwargs):
            print("Fallback comparison not available.")
            return pd.DataFrame(columns=["Copula Family", "Status"])
except ImportError:
    print("WARNING: CopulaComparison module could not be imported")
    # Define a dummy function to prevent crashes

    def compare_copulas(df, **kwargs):
        print("Copula comparison not available.")
        return pd.DataFrame(columns=["Copula Family", "Status"])


# ------------------------------------------------------------------------- #
# Optional external‑package versions versus local fallback implementations
# ------------------------------------------------------------------------- #


def _flex(path: str, fallback: str):
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        return importlib.import_module(fallback)


GARCHVineCopula = _flex("copula.TimeSeries.GARCHVineCopula",
                        "GARCHVineCopula").GARCHVineCopula
DCCCopula = _flex("copula.TimeSeries.DCCCopula",       "DCCCopula").DCCCopula
CoVaRCopula = _flex("copula.TimeSeries.CoVaRCopula",
                    "CoVaRCopula").CoVaRCopula


# ------------------------------------------------------------------------- #
# Helpers
# ------------------------------------------------------------------------- #
def plot_returns(df: pd.DataFrame, out: str = "returns_timeseries.png") -> None:
    n = df.shape[1]
    plt.figure(figsize=(12, 2.5 * n))
    for i, col in enumerate(df.columns, 1):
        plt.subplot(n, 1, i)
        plt.plot(df.index, df[col], lw=.7)
        plt.title(col)
        plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def run_garch_vine(df, alpha=0.05):
    print("\n⚙️  Fitting **GARCH‑Vine Copula** model …")
    res = GARCHVineCopula().fit(df).compute_risk_measures(alpha=alpha)
    print(f"\n===== GARCH‑Vine Risk (α = {alpha}) =====")
    for a, v in res["VaR"].items():
        print(f"VaR[{a}] = {v:.5f},  CVaR = {res['CVaR'][a]:.5f}")
    print(f"Portfolio VaR  = {res['Portfolio_VaR']:.5f}")
    print(f"Portfolio CVaR = {res['Portfolio_CVaR']:.5f}")
    return res


def run_dcc(df, alpha=0.05):
    print("\n⚙️  Fitting **DCC‑GARCH Copula** model …")
    res = DCCCopula().fit(df).compute_risk_measures(alpha=alpha)
    print(f"\n===== DCC‑GARCH Risk (α = {alpha}) =====")
    for a, v in res["VaR"].items():
        print(f"VaR[{a}] = {v:.5f},  CVaR = {res['CVaR'][a]:.5f}")
    print(f"High‑Corr VaR  = {res['High_Corr_VaR']:.5f}")
    print(f"High‑Corr CVaR = {res['High_Corr_CVaR']:.5f}")
    return res


def run_covar(df, alpha=0.05):
    print("\n⚙️  Fitting **CoVaR Copula** model …")
    res = CoVaRCopula().fit(df).compute_risk_measures(
        alpha=alpha, conditioning_assets=list(df.columns)
    )
    print(f"\n===== CoVaR Risk (α = {alpha}) =====")
    for cond in df.columns:
        cd = res[f"CoVaR_{cond}"]
        print(f"\n-- Conditioning on {cond} stress --")
        for tgt, stats in cd.items():
            print(
                f"{tgt:<25s} VaR={stats['VaR']:.5f}  "
                f"CoVaR={stats['CoVaR']:.5f}  ΔCoVaR={stats['DeltaCoVaR']:.5f}"
            )
        print(
            f"Systemic impact (ΣΔCoVaR) = {res[f'Systemic_Impact_{cond}']:.5f}")

    print("\nSystem‑Stress VaR (all conditioning assets stressed):")
    for asset, val in res["System_Stress_VaR"].items():
        print(f"{asset:<25s} {val:.5f}")
    return res


def load_synthetic_data(config=None):
    """
    Load synthetic data based on environment variables or default configuration.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary, if None will read from environment variables
        
    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic returns
    """
    if config is None:
        config = {
            # Default configuration
            "example_dataset": os.getenv("EXAMPLE_DATASET", ""),
            "n_assets": int(os.getenv("N_ASSETS", "3")),
            "n_obs": int(os.getenv("N_OBS", "500")),
            "distribution": os.getenv("DIST", "normal"),
            "df": int(os.getenv("DF", "5")),
            "skew": float(os.getenv("SKEW", "-0.5")),
            "use_garch": os.getenv("USE_GARCH", "0") == "1",
            "random_seed": int(os.getenv("SEED", "42")),
            "correlation_type": os.getenv("CORRELATION", "random")
        }

    print("\n===== SYNTHETIC DATA CONFIGURATION =====")
    for k, v in config.items():
        print(f"{k}: {v}")

    # Check if a predefined example dataset is requested
    if config["example_dataset"]:
        print(
            f"\n🧪 Loading example synthetic dataset: {config['example_dataset']}")
        examples = generate_example_datasets()
        if config["example_dataset"] in examples:
            return examples[config["example_dataset"]]
        else:
            print(f"Example dataset {config['example_dataset']} not found.")
            print(f"Available datasets: {list(examples.keys())}")
            print("Falling back to custom configuration.")

    # Prepare correlation matrix based on configuration
    correlation_matrix = None
    if config["correlation_type"] != "random":
        if config["correlation_type"] == "high":
            # High correlation (0.7-0.9)
            base_corr = np.random.uniform(0.7, 0.9, size=(
                config["n_assets"], config["n_assets"]))
        elif config["correlation_type"] == "medium":
            # Medium correlation (0.3-0.6)
            base_corr = np.random.uniform(0.3, 0.6, size=(
                config["n_assets"], config["n_assets"]))
        elif config["correlation_type"] == "low":
            # Low correlation (-0.2-0.3)
            base_corr = np.random.uniform(-0.2, 0.3,
                                          size=(config["n_assets"], config["n_assets"]))
        elif config["correlation_type"] == "negative":
            # Negative correlation (-0.8--0.2)
            base_corr = np.random.uniform(-0.8, -0.2,
                                          size=(config["n_assets"], config["n_assets"]))
        elif config["correlation_type"] == "mixed":
            # Mixed correlation (-0.5-0.8)
            base_corr = np.random.uniform(-0.5, 0.8,
                                          size=(config["n_assets"], config["n_assets"]))
        else:
            print(f"Unknown correlation type: {config['correlation_type']}")
            print("Using random correlation matrix.")
            base_corr = None

        if base_corr is not None:
            # Make symmetric with 1s on diagonal
            correlation_matrix = np.triu(base_corr, 1)
            correlation_matrix = correlation_matrix + correlation_matrix.T
            np.fill_diagonal(correlation_matrix, 1.0)

            # Ensure it's positive definite (if not, we'll get a random one)
            try:
                np.linalg.cholesky(correlation_matrix)
            except np.linalg.LinAlgError:
                print("Warning: Generated correlation matrix is not positive definite.")
                print("Using spectral decomposition to make it positive definite.")
                eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
                eigvals = np.maximum(eigvals, 1e-8)
                correlation_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
                # Ensure proper correlation matrix properties
                d = np.sqrt(np.diag(correlation_matrix))
                correlation_matrix = correlation_matrix / np.outer(d, d)
                np.fill_diagonal(correlation_matrix, 1.0)

    print("\n🧪 Generating synthetic data with specified parameters...")
    df = generate_synthetic_returns(
        n_assets=config["n_assets"],
        n_obs=config["n_obs"],
        correlation_matrix=correlation_matrix,
        distribution=config["distribution"],
        df=config["df"],
        skew=config["skew"],
        random_seed=config["random_seed"]
    )

    # Add GARCH effects if requested
    if config["use_garch"]:
        print("Adding GARCH volatility clustering effects...")
        df = add_garch_effects(
            df,
            volatility_persistence=float(
                os.getenv("GARCH_PERSISTENCE", "0.95")),
            volatility_of_volatility=float(os.getenv("GARCH_VOL", "0.2"))
        )

    return df


# ------------------------------------------------------------------------- #
# Main driver
# ------------------------------------------------------------------------- #
def main(base_dir="data", alpha=0.05, use_synthetic=False):
    if use_synthetic:
        print(f"🧪 Using synthetic data instead of loading from {base_dir}/")
        df = load_synthetic_data()
    else:
        print(f"🔍 Loading data from: {base_dir}/ …")
        df = load_returns_from_data_folder(base_dir)

    # quick sanity checks
    print("\n===== DATA SUMMARY =====")
    print(df.describe())
    print("\nCorrelation matrix:\n", df.corr().round(4))
    plot_returns(df)

    # risk engines
    run_garch_vine(df, alpha)
    run_dcc(df, alpha)
    run_covar(df, alpha)

    # copula comparison - with error handling
    try:
        print("\nAttempting to run standard copula comparison...")
        comp = compare_copulas(df)
        print("\n===== COPULA COMPARISON TABLE =====")
        pd.set_option("display.width", 150, "display.max_columns", None)
        print(comp.to_string(index=False, float_format=lambda x: f"{x: .6g}"))
        comp.to_csv("copula_comparison.csv", index=False)
        print("\nSaved table to copula_comparison.csv and Q‑Q plots to ./qqplots/.")
    except Exception as e:
        print(f"\nERROR: Standard copula comparison failed: {str(e)}")
        print("Traceback:")
        traceback.print_exc(file=sys.stdout)

        # Try the fallback implementation
        try:
            print("\nAttempting fallback copula comparison...")
            comp = compare_copulas_fallback(df)
            print("\n===== FALLBACK COPULA COMPARISON TABLE =====")
            pd.set_option("display.width", 150, "display.max_columns", None)
            print(comp.to_string(index=False,
                  float_format=lambda x: f"{x: .6g}"))
            comp.to_csv("copula_comparison_fallback.csv", index=False)
            print("\nSaved fallback table to copula_comparison_fallback.csv.")
        except Exception as e2:
            print(f"\nERROR: Fallback comparison also failed: {str(e2)}")
            print("Continuing without copula comparison.")


if __name__ == "__main__":
    use_synthetic = os.getenv("USE_SYNTHETIC", "0") == "1"
    main(
        os.getenv("DATA_DIR", "data"),
        float(os.getenv("ALPHA", "0.05")),
        use_synthetic
    )
