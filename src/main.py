import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skewnorm, t
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional, Union



# Example usage
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


def main():
    """Main function to demonstrate copula modeling."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random dataset
    print("Generating random dataset with copula model...")
    data = create_random_dataset(n_samples=2000)

    # Display basic information
    print("\nDataset Information:")
    print(f"Number of samples: {len(data)}")
    print(f"Variables: {', '.join(data.columns)}")

    print("\nSummary Statistics:")
    print(data.describe())

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
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.title('Joint Distribution: Stock Returns vs Volatility Index', fontsize=14)
    sns.jointplot(data=data, x='stock_returns',
                  y='volatility_index', kind='kde', fill=True)
    plt.tight_layout()
    plt.show()

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
    plt.show()


if __name__ == "__main__":
    main()
