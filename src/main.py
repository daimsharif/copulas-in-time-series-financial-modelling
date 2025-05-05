"""
Created on 05/05/2025

@author: Aryan

Filename: combined_copula_script.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import from general copula modules
from copula.CopulaDistribution import CopulaDistribution
from copula.CopulaModel import CopulaModel
from copula.Marginal import Marginal
from copula.DataAnalyzer import DataAnalyzer
from copula.StudentTCopula import StudentTCopula
from copula.ClaytonCopula import ClaytonCopula

# Import from time series copula modules
from copula.TimeSeries.CoVaRCopula import CoVaRCopula
from copula.TimeSeries.DCCCopula import DCCCopula
from copula.TimeSeries.GARCHVineCopula import GARCHVineCopula


def plot_initial_timeseries(returns_df: pd.DataFrame):
    """
    Plot the initial synthetic financial time series data as separate plots for each asset.

    Args:
        returns_df: DataFrame with financial time series data
    """
    n_assets = returns_df.shape[1]
    plt.figure(figsize=(12, 3 * n_assets))

    for i, col in enumerate(returns_df.columns, 1):
        plt.subplot(n_assets, 1, i)
        plt.plot(returns_df.index, returns_df[col], label=col, linewidth=1)
        plt.title(f"{col} Returns")
        plt.xlabel("Time Index")
        plt.ylabel("Returns")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    plt.savefig("initial_timeseries_per_asset.png")
    # plt.show()

def create_random_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Create a random dataset with interesting dependence structures.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with the generated data
    """
    # Create marginal distributions
    marginals = [
        Marginal("stock_returns", stats.norm, {"loc": 0.05, "scale": 0.2}),
        Marginal("bond_yields", stats.skewnorm, {
                 "a": 4, "loc": 0.03, "scale": 0.01}),
        Marginal("commodity_prices", stats.t, {
                 "df": 3, "loc": 50, "scale": 10}),
        Marginal("volatility_index", stats.lognorm,
                 {"s": 0.5, "loc": 0, "scale": 0.2})
    ]

    # Create correlation matrix with some interesting structure
    corr_matrix = np.array([
        [1.0, -0.3,  0.2,  0.6],
        [-0.3,  1.0, -0.1, -0.4],
        [0.2, -0.1,  1.0,  0.1],
        [0.6, -0.4,  0.1,  1.0]
    ])

    # Create Student's t copula model
    t_copula = StudentTCopula()
    copula_model = CopulaModel(t_copula, marginals, dimension=4)

    # Generate data
    data = copula_model.simulate(
        n_samples, {'corr_matrix': corr_matrix, 'df': 4})

    return data


def generate_financial_timeseries(n_obs: int = 1000, n_assets: int = 4) -> pd.DataFrame:
    """
    Generate synthetic financial time series data with correlation structure
    and volatility clustering.
    
    Args:
        n_obs: Number of observations (days)
        n_assets: Number of assets
        
    Returns:
        DataFrame with generated return series
    """
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

    return returns_df


def analyze_copula_structure(data: pd.DataFrame):
    """
    Analyze the copula structure of the given data.
    
    Args:
        data: DataFrame containing variables to analyze
    """
    # Analyze data
    analyzer = DataAnalyzer(data)

    # Compute correlations
    correlations = analyzer.compute_correlations()
    print("\nPearson Correlation Matrix:")
    print(correlations['Pearson'])

    print("\nSpearman Correlation Matrix:")
    print(correlations['Spearman'])

    # Compute tail dependence
    tail_dep = analyzer.compute_tail_dependence(
        'stock_returns', 'volatility_index')
    print("\nTail Dependence between Stock Returns and Volatility Index:")
    print(f"Upper tail dependence: {tail_dep['upper_tail_dependence']:.4f}")
    print(f"Lower tail dependence: {tail_dep['lower_tail_dependence']:.4f}")

    # Create visualizations
    plt.figure(figsize=(12, 10))
    plt.suptitle('Scatter Matrix Plot of Financial Variables', fontsize=16)
    sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.tight_layout()
    plt.savefig('scatter_matrix.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title('Joint Distribution: Stock Returns vs Volatility Index', fontsize=14)
    sns.jointplot(data=data, x='stock_returns',
                  y='volatility_index', kind='kde', fill=True)
    plt.tight_layout()
    plt.savefig('joint_distribution.png')
    plt.close()

    # Example of bivariate Clayton copula
    print("\nGenerating samples from a bivariate Clayton copula...")
    clayton = ClaytonCopula()
    clayton_samples = clayton.simulate(1000, {'theta': 3.0})

    plt.figure(figsize=(8, 6))
    plt.scatter(clayton_samples[:, 0], clayton_samples[:, 1], alpha=0.5)
    plt.title('Bivariate Clayton Copula Samples', fontsize=14)
    plt.xlabel('U1')
    plt.ylabel('U2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clayton_samples.png')
    plt.close()


def analyze_financial_risk(returns_df: pd.DataFrame):
    """
    Analyze financial risk using various copula models.
    
    Args:
        returns_df: DataFrame with financial time series data
    """
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
    print(pd.DataFrame(recent_corr,
                       index=returns_df.columns,
                       columns=returns_df.columns).round(3))

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


def combined_copula_analysis():
    """Main function combining both general and time series copula analysis."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Part 1: General Copula Analysis with random dataset
    print("\n===== PART 1: GENERAL COPULA ANALYSIS =====")
    print("Generating random dataset with copula model...")
    data = create_random_dataset(n_samples=2000)

    # Display basic information
    print("\nDataset Information:")
    print(f"Number of samples: {len(data)}")
    print(f"Variables: {', '.join(data.columns)}")

    print("\nSummary Statistics:")
    print(data.describe())

    # Analyze copula structure
    analyze_copula_structure(data)

    # Part 2: Financial Time Series Copula Analysis
    print("\n\n===== PART 2: FINANCIAL TIME SERIES COPULA ANALYSIS =====")
    print("Generating synthetic financial time series data...")
    returns_df = generate_financial_timeseries(n_obs=1000, n_assets=4)

    # ⬇️ Insert this here ⬇️
    plot_initial_timeseries(returns_df)

    print("\nReturns Data Information:")
    print(f"Number of time periods: {len(returns_df)}")
    print(f"Assets: {', '.join(returns_df.columns)}")

    print("\nSummary Statistics of Returns:")
    print(returns_df.describe())

    # Plot time series data
    plt.figure(figsize=(12, 8))
    plt.plot(returns_df.index, returns_df)
    plt.title('Simulated Financial Asset Returns', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend(returns_df.columns)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('financial_returns.png')
    plt.close()

    # Analyze financial risk
    analyze_financial_risk(returns_df)


if __name__ == "__main__":
    combined_copula_analysis()
