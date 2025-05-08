"""
Utility to generate synthetic market data with specified correlation structure.

This module provides functions to create simulated return data with:
- Customizable correlation matrices
- Various marginal distributions (normal, student-t, etc.)
- Optional time series features (autocorrelation, volatility clustering)

Can be used as an alternative to real data for testing statistical models.

Author: Added by user (2025-05-08)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union, Literal


def generate_synthetic_returns(
    n_assets: int = 3,
    n_obs: int = 500,
    correlation_matrix: Optional[np.ndarray] = None,
    mean_returns: Optional[np.ndarray] = None,
    std_returns: Optional[np.ndarray] = None,
    asset_names: Optional[List[str]] = None,
    distribution: Literal["normal", "t", "skewed_t"] = "normal",
    df: int = 5,  # degrees of freedom for t-distribution
    skew: float = -0.5,  # skewness parameter for skewed distributions
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic return data with specified correlation structure.
    
    Parameters
    ----------
    n_assets : int
        Number of assets to simulate
    n_obs : int
        Number of observations (time steps)
    correlation_matrix : np.ndarray, optional
        Correlation matrix (must be positive definite). If None, a random 
        correlation matrix will be generated.
    mean_returns : np.ndarray, optional
        Mean returns for each asset. If None, random values around 0 will be used.
    std_returns : np.ndarray, optional
        Standard deviations for each asset. If None, random values will be used.
    asset_names : List[str], optional
        Names for the assets. If None, names like 'Asset_1', 'Asset_2', etc. will be used.
    distribution : str
        Distribution type: 'normal', 't', or 'skewed_t'
    df : int
        Degrees of freedom for t-distribution
    skew : float
        Skewness parameter for skewed distributions
    random_seed : int, optional
        Seed for random number generation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic returns, where columns are assets and rows are time steps
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random correlation matrix if not provided
    if correlation_matrix is None:
        # Create a random correlation matrix that is guaranteed to be positive definite
        A = np.random.randn(n_assets, n_assets)
        B = np.dot(A, A.T)  # This gives a positive semi-definite matrix

        # Convert to correlation matrix
        D = np.sqrt(np.diag(1 / np.diag(B)))
        correlation_matrix = np.dot(np.dot(D, B), D)
    else:
        # Verify dimensions
        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"Correlation matrix must be {n_assets}x{n_assets}")

        # Check if the matrix is approximately symmetric
        if not np.allclose(correlation_matrix, correlation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")

        # Check diagonal elements
        if not np.allclose(np.diag(correlation_matrix), np.ones(n_assets)):
            raise ValueError(
                "Diagonal elements of correlation matrix must be 1")

    # Generate random means and standard deviations if not provided
    if mean_returns is None:
        mean_returns = np.random.uniform(-0.001, 0.001, n_assets)

    if std_returns is None:
        std_returns = np.random.uniform(0.01, 0.03, n_assets)

    # Generate correlated normal random variables
    try:
        # Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)
        Z = np.random.randn(n_obs, n_assets)
        X = Z @ L.T  # Correlated standard normal variables
    except np.linalg.LinAlgError:
        # Fallback if matrix is not positive definite
        print("Warning: Correlation matrix is not positive definite. Using eigenvalue decomposition.")
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-8)  # Ensure positive eigenvalues
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        Z = np.random.randn(n_obs, n_assets)
        X = Z @ L.T

    # Transform to desired distribution
    if distribution == "normal":
        Y = X
    elif distribution == "t":
        # Convert normal to t-distribution
        u = stats.chi2.rvs(df, size=(n_obs, 1))
        Y = X * np.sqrt(df / u)
    elif distribution == "skewed_t":
        # Basic skewed t implementation (simplified)
        u = stats.chi2.rvs(df, size=(n_obs, 1))
        Y = X * np.sqrt(df / u)
        # Apply skewness transformation
        delta = skew / np.sqrt(1 + skew**2)
        Y = delta * np.abs(Y) + np.sqrt(1 - delta**2) * Y
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Scale by standard deviations and add means
    returns = Y * std_returns + mean_returns

    # Create DataFrame
    if asset_names is None:
        asset_names = [f"Asset_{i+1}" for i in range(n_assets)]

    # Create date index (daily business days)
    end_date = pd.Timestamp.today()
    # Extra days to account for weekends
    start_date = end_date - pd.Timedelta(days=n_obs * 2)
    date_index = pd.date_range(
        start=start_date, end=end_date, freq='B')[:n_obs]

    df = pd.DataFrame(returns, index=date_index, columns=asset_names)

    return df


def add_garch_effects(
    returns: pd.DataFrame,
    garch_params: Optional[Dict[str, Tuple[float, float, float]]] = None,
    volatility_persistence: float = 0.95,
    volatility_of_volatility: float = 0.2,
) -> pd.DataFrame:
    """
    Add GARCH-like volatility clustering effects to returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Original returns data
    garch_params : Dict[str, Tuple[float, float, float]], optional
        Dictionary with asset names as keys and (omega, alpha, beta) GARCH parameters as values
        If None, parameters will be randomly generated with specified persistence
    volatility_persistence : float
        Target level of volatility persistence (alpha + beta) if garch_params is None
    volatility_of_volatility : float
        Controls the amplitude of volatility fluctuations
        
    Returns
    -------
    pd.DataFrame
        Returns with GARCH effects
    """
    result = returns.copy()
    n_obs = len(returns)

    for col in returns.columns:
        # Use provided parameters or generate random ones
        if garch_params is not None and col in garch_params:
            omega, alpha, beta = garch_params[col]
        else:
            # Generate random parameters with target persistence
            beta = np.random.uniform(
                volatility_persistence - 0.2, volatility_persistence)
            alpha = volatility_persistence - beta
            omega = (1 - alpha - beta) * returns[col].var()

        # Initialize volatility process
        sigma2 = np.zeros(n_obs)
        sigma2[0] = returns[col].var()

        # Simulate GARCH process
        for t in range(1, n_obs):
            sigma2[t] = omega + alpha * returns.iloc[t-1,
                                                     returns.columns.get_loc(col)]**2 + beta * sigma2[t-1]

        # Apply volatility to returns
        # Preserve the correlation structure by scaling the returns
        orig_std = returns[col].std()
        time_varying_std = np.sqrt(sigma2)
        result[col] = returns[col] * (time_varying_std / orig_std)

    return result


def generate_example_datasets() -> Dict[str, pd.DataFrame]:
    """
    Generate a few example datasets with different characteristics.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of named example datasets
    """
    examples = {}

    # Example 1: Three highly correlated assets (equity-like)
    corr1 = np.array([
        [1.0, 0.7, 0.8],
        [0.7, 1.0, 0.75],
        [0.8, 0.75, 1.0]
    ])
    examples["highly_correlated"] = generate_synthetic_returns(
        n_assets=3,
        correlation_matrix=corr1,
        std_returns=np.array([0.015, 0.02, 0.025]),
        asset_names=["Stock_A", "Stock_B", "Stock_C"],
        distribution="skewed_t",
        random_seed=42
    )

    # Example 2: Mixed asset classes
    corr2 = np.array([
        [1.0, 0.6, 0.3, -0.2, 0.1],
        [0.6, 1.0, 0.2, -0.1, 0.2],
        [0.3, 0.2, 1.0, 0.1, 0.4],
        [-0.2, -0.1, 0.1, 1.0, -0.3],
        [0.1, 0.2, 0.4, -0.3, 1.0]
    ])
    examples["mixed_assets"] = generate_synthetic_returns(
        n_assets=5,
        correlation_matrix=corr2,
        std_returns=np.array([0.018, 0.022, 0.015, 0.008, 0.012]),
        asset_names=["Equity", "Real_Estate",
                     "Corp_Bonds", "Govt_Bonds", "Commodities"],
        distribution="t",
        df=6,
        random_seed=123
    )

    # Example 3: Low correlation assets with GARCH effects
    corr3 = np.array([
        [1.0, 0.2, -0.1, 0.15],
        [0.2, 1.0, 0.25, -0.05],
        [-0.1, 0.25, 1.0, 0.1],
        [0.15, -0.05, 0.1, 1.0]
    ])
    base_returns = generate_synthetic_returns(
        n_assets=4,
        correlation_matrix=corr3,
        asset_names=["Crypto", "Tech", "Utilities", "Gold"],
        std_returns=np.array([0.03, 0.02, 0.01, 0.015]),
        random_seed=456
    )
    examples["garch_effects"] = add_garch_effects(
        base_returns,
        volatility_persistence=0.98,
        volatility_of_volatility=0.3
    )

    return examples


if __name__ == "__main__":
    # Demo code
    print("Generating example synthetic returns datasets...")

    # Custom correlation matrix
    print("\nGenerating custom correlated returns...")
    custom_corr = np.array([
        [1.0, 0.5, -0.3],
        [0.5, 1.0, 0.2],
        [-0.3, 0.2, 1.0]
    ])
    custom_data = generate_synthetic_returns(
        correlation_matrix=custom_corr,
        asset_names=["Asset_A", "Asset_B", "Asset_C"],
        n_obs=1000,
        distribution="t",
        df=5,
        random_seed=42
    )

    print("Generated data shape:", custom_data.shape)
    print("Sample of generated data:")
    print(custom_data.head())

    print("\nCorrelation matrix of generated data:")
    print(custom_data.corr().round(2))

    print("\nDescriptive statistics:")
    print(custom_data.describe().round(4))
