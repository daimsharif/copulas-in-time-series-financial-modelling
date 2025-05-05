"""
Created on 05/05/2025

@author: Aryan

Filename: main.py

Relative Path: src/copula/TimeSeries/main.py
"""

import numpy as np
import pandas as pd


from copula.TimeSeries.CoVaRCopula import CoVaRCopula
from copula.TimeSeries.DCCCopula import DCCCopula
from copula.TimeSeries.GARCHVineCopula import GARCHVineCopula





def main():
    """Example usage of the financial copula models."""
    # Generate synthetic data
    np.random.seed(42)
    n_obs = 1000
    n_assets = 4

    # Asset names
    asset_names = ['Market_Index', 'Bank_A', 'Bank_B', 'Insurer_C']

    # Generate correlated returns with some interesting properties
    mu = np.array([0.05, 0.03, 0.04, 0.02]) / 252  # Daily expected returns
    sigma = np.array([0.15, 0.20, 0.18, 0.12]) / \
        np.sqrt(252)  # Daily volatilities

    # Correlation matrix with stronger correlations during stress
    corr_matrix = np.array([
        [1.00, 0.50, 0.45, 0.30],
        [0.50, 1.00, 0.65, 0.25],
        [0.45, 0.65, 1.00, 0.35],
        [0.30, 0.25, 0.35, 1.00]
    ])

    # Covariance matrix
    cov_matrix = np.outer(sigma, sigma) * corr_matrix

    # Generate base returns
    returns = np.random.multivariate_normal(mu, cov_matrix, n_obs)

    # Add some volatility clustering
    for i in range(n_assets):
        vol_cluster = np.zeros(n_obs)
        vol_cluster[0] = 1.0
        for t in range(1, n_obs):
            vol_cluster[t] = 0.95 * vol_cluster[t-1] + \
                0.05 * np.random.normal(0, 1)

        # Apply volatility clustering
        volatility_factor = np.exp(0.2 * vol_cluster)
        returns[:, i] *= volatility_factor

    # Create a stress period
    stress_period = slice(400, 500)
    # Double volatility and increase correlation during stress
    stress_factor = 2.5
    returns[stress_period] *= stress_factor

    # Convert to DataFrame
    returns_df = pd.DataFrame(returns, columns=asset_names)

    # Example 1: GARCH-Vine Copula Model
    print("\n1. GARCH-based Vine Copula Model")
    print("================================")

    garch_vine_model = GARCHVineCopula()
    garch_vine_model.fit(returns_df)

    # Simulate from the model
    simulated_returns = garch_vine_model.simulate(n_samples=1000)

    # Compute risk measures
    risk_measures = garch_vine_model.compute_risk_measures(alpha=0.05)

    print("Value-at-Risk (95%):")
    for asset, var in risk_measures['VaR'].items():
        print(f"  {asset}: {var:.4f}")

    print("\nPortfolio VaR (95%): {:.4f}".format(
        risk_measures['Portfolio_VaR']))
    print("Portfolio CVaR (95%): {:.4f}".format(
        risk_measures['Portfolio_CVaR']))

    # Example 2: DCC Copula Model
    print("\n2. Dynamic Conditional Correlation Copula Model")
    print("===============================================")

    dcc_model = DCCCopula()
    dcc_model.fit(returns_df, dcc_params={'a': 0.03, 'b': 0.95})

    # Simulate from the model
    simulated_returns_dcc = dcc_model.simulate(n_samples=1000)

    print("DCC Model Correlation Dynamics:")
    recent_corr = dcc_model.correlation_matrices[-1]
    print(pd.DataFrame(recent_corr, index=asset_names, columns=asset_names).round(3))

    # Example 3: CoVaR Copula Model
    print("\n3. CoVaR Copula Model")
    print("====================")

    covar_model = CoVaRCopula()
    covar_model.fit(returns_df, copula_type='t', copula_params={'df': 4})

    # Compute CoVaR with Market_Index as the conditioning asset
    covar_results = covar_model.compute_covar(
        conditioning_asset='Market_Index',
        alpha=0.05
    )

    print("CoVaR Results (Market_Index in distress):")
    for asset, measures in covar_results.items():
        print(f"\n{asset}:")
        print(f"  VaR (95%): {measures['VaR']:.4f}")
        print(f"  CoVaR (95%): {measures['CoVaR']:.4f}")
        print(f"  ΔCoVaR: {measures['DeltaCoVaR']:.4f}")

    # Compute systemic risk measures
    sys_risk = covar_model.compute_risk_measures(
        conditioning_assets=['Market_Index', 'Bank_A']
    )

    print("\nSystemic Impact:")
    for asset in ['Market_Index', 'Bank_A']:
        print(f"  {asset}: {sys_risk[f'Systemic_Impact_{asset}']:.4f}")


if __name__ == "__main__":
    main()
