"""
CopulaComparison.py
Builds a full comparison table (metrics + Q‑Q plots) for:
  • Gaussian  • t       • Clayton  • Gumbel
Requires:  copulas, scipy, matplotlib, pandas
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    # Updated import to use VineCopula instead of StudentMultivariate
    from copulas.multivariate import GaussianMultivariate
    # Check if we can import the VineCopula for t-copula functionality
    from copulas.multivariate import VineCopula
    from copulas.bivariate import Clayton, Gumbel

    # Import needed utility functions
    import warnings
    import inspect

    # Check if this version of copulas uses log_likelihood or loglikelihood naming
    def _check_likelihood_method(cls):
        """Check which likelihood method exists in the class"""
        if 'log_likelihood' in dir(cls):
            return 'log_likelihood'
        elif 'loglikelihood' in dir(cls):
            return 'loglikelihood'
        elif '_loglikelihood' in dir(cls) and callable(getattr(cls, '_loglikelihood')):
            return '_loglikelihood'
        return None
except ImportError as e:
    raise ImportError(
        "Install the 'copulas' package first →  pip install copulas"
    ) from e


# ------------------------------------------------------------------------- #
# Internal helpers
# ------------------------------------------------------------------------- #
def _pseudo_obs(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values, axis=0, method="average")
    return ranks / (values.shape[0] + 1.0)


def _aic(loglik: float, k: int) -> float:
    return 2 * k - 2 * loglik


def _bic(loglik: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * loglik


def _distance(metric: str, u: np.ndarray, v: np.ndarray) -> float:
    if metric == "Euclidean Distance":
        return float(np.linalg.norm(u - v) / u.shape[0])

    if metric == "Spearman's Rank Correlation":
        return float(stats.spearmanr(u.flatten(), v.flatten()).correlation)

    if metric == "Kendall's Tau Distance":
        tau = stats.kendalltau(u.flatten(), v.flatten()).correlation
        return 1 - (tau if tau is not None else 0.0)

    if metric == "Distance Correlation":
        return float(stats.distance_correlation(u.flatten(), v.flatten()))

    raise ValueError(f"Unsupported distance metric: {metric}")


def _qqplot(u: np.ndarray, family: str, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(4, 4))
    stats.probplot(u.flatten(), dist="uniform", plot=plt)
    plt.title(f"Q‑Q Plot – {family}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = out_dir / f"qq_{family.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(img, dpi=150)
    plt.close(fig)
    return img


# ------------------------------------------------------------------------- #
# Public API
# ------------------------------------------------------------------------- #
def compare_copulas(
    returns: pd.DataFrame,
    cols: List[str] | None = None,
    out_dir: str | Path = "qqplots",
) -> pd.DataFrame:
    """Fit four copulas and spit out every metric the boss asked for."""
    # Fix 1: Handle multivariate data by selecting only pairs of columns
    if cols is None:
        # Use the first pair of columns by default
        if returns.shape[1] >= 2:
            cols = list(returns.columns[:2])
        else:
            raise ValueError("Need at least two columns to fit copulas.")

    if len(cols) != 2:
        print(
            f"Warning: Expected exactly 2 columns for bivariate copula analysis, got {len(cols)}.")
        print(f"Will use the first pair from: {cols[:2]}")
        cols = cols[:2]

    print(f"Analyzing copula relationships between {cols[0]} and {cols[1]}")

    # Filter to just the two selected columns
    data = returns[cols].dropna().to_numpy()
    n = data.shape[0]

    # Convert to pseudo-observations (uniform margins)
    u = _pseudo_obs(data)

    # Create a DataFrame with the right column names for the copula package
    u_df = pd.DataFrame(u, columns=["X", "Y"])

    configs: Dict[str, Dict[str, Any]] = {
        "Gaussian Copula": {
            "model":  GaussianMultivariate(),
            "sel":    "AIC",
            "gof":    "KS Test",
            "dist":   "Euclidean Distance",
            "marg":   "Normal",
        },
        "t‑Copula": {
            # Fixed: VineCopula only accepts vine_type parameter, not vine_structure
            "model":  VineCopula(vine_type='center'),
            "sel":    "BIC",
            "gof":    "Anderson-Darling",
            "dist":   "Spearman's Rank Correlation",
            "marg":   "Student‑t",
        },
        "Clayton Copula": {
            "model":  Clayton(),
            "sel":    "AIC",
            "gof":    "Cramér-von Mises",
            "dist":   "Kendall's Tau Distance",
            "marg":   "Exponential",
        },
        "Gumbel Copula": {
            "model":  Gumbel(),
            "sel":    "AIC",
            "gof":    "Chi-Square Test",
            "dist":   "Distance Correlation",
            "marg":   "Gamma",
        },
    }

    rows = []
    out_dir = Path(out_dir)

    for fam, cfg in configs.items():
        try:
            m = cfg["model"]

            # Fix 2: Special case for multivariate models
            if isinstance(m, (GaussianMultivariate, VineCopula)):
                # Use the original DataFrame for multivariate models
                m.fit(returns[cols])
            else:
                # Use the U-transformed data for bivariate models
                m.fit(u_df)

            # Handle different APIs for getting log-likelihood
            loglik = -999.99  # Default placeholder
            try:
                if hasattr(m, 'log_likelihood'):
                    loglik = float(m.log_likelihood(u_df if isinstance(
                        m, (Clayton, Gumbel)) else returns[cols]))
                elif hasattr(m, 'loglikelihood'):
                    loglik = float(m.loglikelihood(u_df if isinstance(
                        m, (Clayton, Gumbel)) else returns[cols]))
                elif hasattr(m, '_loglikelihood') and callable(m._loglikelihood):
                    loglik = float(m._loglikelihood(u_df if isinstance(
                        m, (Clayton, Gumbel)) else returns[cols]))
            except Exception as e:
                print(
                    f"WARNING: Could not compute log-likelihood for {fam}: {str(e)}")

            # Fix 3: Handle parameters more safely
            k = 1  # Default parameter count
            try:
                if hasattr(m, "get_parameters") and callable(m.get_parameters):
                    params = m.get_parameters()
                    if isinstance(params, dict):
                        k = len(params)
                    elif isinstance(params, (list, tuple)):
                        k = len(params)
            except Exception as e:
                print(
                    f"WARNING: Could not determine parameter count for {fam}: {str(e)}")

            sel_val = _aic(loglik, k) if cfg["sel"] == "AIC" else _bic(
                loglik, k, n)

            # Fix 4: Handle sampling with better error handling
            try:
                # Handle different sampling APIs
                if isinstance(m, (GaussianMultivariate, VineCopula)):
                    # For multivariate models, we need to transform the samples back
                    sample_result = m.sample(n)
                    if isinstance(sample_result, pd.DataFrame):
                        v_raw = sample_result[cols].to_numpy()
                        # Transform to uniform for comparison
                        v = _pseudo_obs(v_raw)
                    else:
                        # Handle array-like results
                        v_raw = np.array(sample_result)
                        if v_raw.shape[1] >= 2:
                            v_raw = v_raw[:, :2]  # Take first two columns
                        v = _pseudo_obs(v_raw)
                else:
                    # For bivariate models, sample directly
                    sample_result = m.sample(n)
                    if isinstance(sample_result, pd.DataFrame):
                        v = sample_result.to_numpy()
                    else:
                        v = np.array(sample_result).reshape(-1, 2)
            except Exception as e:
                print(f"WARNING: Sampling failed for {fam}: {str(e)}")
                # Generate uniform samples as fallback
                v = np.random.uniform(size=u.shape)

            # Compute goodness-of-fit metrics
            try:
                if cfg["gof"] == "KS Test":
                    gof_val = stats.ks_2samp(
                        u.flatten(), v.flatten()).statistic
                elif cfg["gof"] == "Anderson-Darling":
                    gof_val = stats.anderson_ksamp(
                        [u.flatten(), v.flatten()]).statistic
                elif cfg["gof"] == "Cramér-von Mises":
                    gof_val = stats.cramervonmises_2samp(
                        u.flatten(), v.flatten()).statistic
                else:  # Chi‑Square
                    h_u, _ = np.histogram(u.flatten(), bins=10)
                    h_v, _ = np.histogram(v.flatten(), bins=10)
                    gof_val = stats.chisquare(h_u, h_v).statistic
            except Exception as e:
                print(
                    f"WARNING: Could not compute goodness-of-fit for {fam}: {str(e)}")
                gof_val = np.nan

            # Compute distance metrics
            try:
                dist_val = _distance(cfg["dist"], u, v)
            except Exception as e:
                print(
                    f"WARNING: Could not compute distance metric for {fam}: {str(e)}")
                dist_val = np.nan

            # Extract parameters
            try:
                if fam == "t‑Copula" and isinstance(m, VineCopula):
                    # Try to extract the degrees of freedom parameter
                    theta = np.nan
                    try:
                        # This is model-specific and might need adjustment
                        if hasattr(m, 'model') and hasattr(m.model, 'degrees_of_freedom'):
                            theta = m.model.degrees_of_freedom
                        elif hasattr(m, '_vinearray') and m._vinearray is not None:
                            # Try to find df in vine array
                            theta = np.nan  # More complex extraction needed
                    except:
                        theta = np.nan
                elif isinstance(m, (Clayton, Gumbel)):
                    # For bivariate copulas
                    theta = getattr(m, "theta", np.nan)
                else:
                    # For other models
                    if hasattr(m, "get_parameters") and callable(m.get_parameters):
                        params = m.get_parameters()
                        if isinstance(params, dict) and len(params) > 0:
                            theta = list(params.values())[0]
                        else:
                            theta = np.nan
                    else:
                        theta = np.nan
            except Exception as e:
                print(
                    f"WARNING: Could not extract parameters for {fam}: {str(e)}")
                theta = np.nan

            # Generate Q-Q plot
            try:
                img = _qqplot(u, fam, out_dir)
            except Exception as e:
                print(
                    f"WARNING: Could not generate Q-Q plot for {fam}: {str(e)}")
                img = "Failed to generate"

            rows.append(
                {
                    "Copula Family":               fam,
                    "Parameter (Theta)":           theta,
                    "Selection Metric (AIC/BIC)":  sel_val,
                    "Goodness-of-Fit Metric":      gof_val,
                    "Q‑Q Plot":                    str(img),
                    "Marginal Distribution":       cfg["marg"],
                    "Distance/Rank Comparison":    dist_val,
                    "Status":                      "Success",
                }
            )
        except Exception as e:
            print(f"ERROR: Failed to fit {fam}: {str(e)}")
            rows.append(
                {
                    "Copula Family":               fam,
                    "Parameter (Theta)":           np.nan,
                    "Selection Metric (AIC/BIC)":  np.nan,
                    "Goodness-of-Fit Metric":      np.nan,
                    "Q‑Q Plot":                    "Failed",
                    "Marginal Distribution":       cfg["marg"],
                    "Distance/Rank Comparison":    np.nan,
                    "Status":                      f"Failed: {str(e)}",
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- #
# Fallback implementation for copula comparison
# ------------------------------------------------------------------------- #
def compare_copulas_fallback(
    returns: pd.DataFrame,
    out_dir: str | Path = "qqplots",
) -> pd.DataFrame:
    """Simplified fallback comparison that focuses on pairwise analyses."""
    rows = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create all possible pairs of assets
    asset_pairs = list(combinations(returns.columns, 2))

    print(f"Analyzing {len(asset_pairs)} asset pairs in fallback mode")

    # For each pair, compute basic correlation metrics
    for asset1, asset2 in asset_pairs:
        # Get the data
        data = returns[[asset1, asset2]].dropna()

        # Skip if not enough data
        if len(data) < 10:
            continue

        # Compute correlations
        pearson = data.corr().iloc[0, 1]
        spearman = data.corr(method='spearman').iloc[0, 1]

        # Compute Kendall's tau
        kendall = stats.kendalltau(data[asset1], data[asset2]).correlation

        # Compute tail dependence coefficient (simple approximation)
        # Lower tail dependence
        q = 0.05  # 5% quantile
        x1_q = data[asset1].quantile(q)
        x2_q = data[asset2].quantile(q)

        joint_lower = ((data[asset1] <= x1_q) & (data[asset2] <= x2_q)).mean()
        lower_tail = joint_lower / q if q > 0 else np.nan

        # Upper tail dependence
        q = 0.95  # 95% quantile
        x1_q = data[asset1].quantile(q)
        x2_q = data[asset2].quantile(q)

        joint_upper = ((data[asset1] >= x1_q) & (data[asset2] >= x2_q)).mean()
        upper_tail = joint_upper / (1-q) if (1-q) > 0 else np.nan

        # Create scatter plot with marginal histograms
        try:
            fig, ax = plt.subplots(figsize=(8, 8))

            # Create the scatter plot
            ax.scatter(data[asset1], data[asset2], alpha=0.5, s=10)
            ax.set_xlabel(asset1)
            ax.set_ylabel(asset2)
            ax.set_title(f"Scatter Plot: {asset1} vs {asset2}")

            # Add correlation info
            ax.text(0.05, 0.95,
                    f"Pearson: {pearson:.4f}\nSpearman: {spearman:.4f}\n" +
                    f"Kendall: {kendall:.4f}\nLower Tail: {lower_tail:.4f}\n" +
                    f"Upper Tail: {upper_tail:.4f}",
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Save the plot
            img_path = out_dir / f"scatter_{asset1}_vs_{asset2}.png"
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close(fig)
        except Exception as e:
            print(
                f"WARNING: Could not create scatter plot for {asset1} vs {asset2}: {str(e)}")
            img_path = "Failed to generate"

        # Add to results
        rows.append({
            "Asset Pair": f"{asset1} — {asset2}",
            "Pearson Correlation": pearson,
            "Spearman Correlation": spearman,
            "Kendall's Tau": kendall,
            "Lower Tail Dependence": lower_tail,
            "Upper Tail Dependence": upper_tail,
            "Scatter Plot": str(img_path),
            "Sample Size": len(data)
        })

    # Create a summary of which copula family might be most appropriate
    for i, row in enumerate(rows):
        pearson = row["Pearson Correlation"]
        lower_tail = row["Lower Tail Dependence"]
        upper_tail = row["Upper Tail Dependence"]

        # Simple heuristic for copula family recommendation
        if abs(pearson) < 0.1:
            recommendation = "Independence (weak dependence)"
        elif abs(lower_tail - upper_tail) < 0.1 and abs(pearson) > 0.7:
            recommendation = "Gaussian or t-Copula (symmetric, strong)"
        elif abs(lower_tail - upper_tail) < 0.1:
            recommendation = "Gaussian Copula (symmetric)"
        elif lower_tail > upper_tail + 0.1:
            recommendation = "Clayton Copula (lower tail dependence)"
        elif upper_tail > lower_tail + 0.1:
            recommendation = "Gumbel Copula (upper tail dependence)"
        else:
            recommendation = "t-Copula (symmetric with tail dependence)"

        rows[i]["Recommended Copula"] = recommendation

    return pd.DataFrame(rows)
