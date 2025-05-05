"""
Created on 05/05/2025

@author: Aryan

Filename: CoVaRCopula.py

Relative Path: src/copula/TimeSeries/CoVaRCopula.py
"""

from typing import Dict, List
from matplotlib import pyplot as plt
from matplotlib.pylab import norm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t
from copula.TimeSeries.FinancialCopulaModel import FinancialCopulaModel
from copula.TimeSeries.GARCHModel import GARCHModel

class CoVaRCopula(FinancialCopulaModel):
    """
    Conditional Value-at-Risk (CoVaR) Copula model for capturing tail risk
    and conditional dependence in systemic risk analysis and stress testing.
    """

    def __init__(self):
        """Initialize the CoVaR Copula model."""
        super().__init__(
            name="CoVaR Copula",
            description="Tail risk & conditional dependence",
            use_case="Systemic risk & stress testing"
        )
        self.garch_models = []
        self.asset_names = None
        self.standardized_residuals = None
        self.copula = None
        self.tail_dependence_matrix = None

    def fit(self, data: pd.DataFrame, **kwargs) -> 'CoVaRCopula':
        """
        Fit the CoVaR Copula model to financial data.
        
        Args:
            data: Financial time series returns
            **kwargs: Additional parameters for fitting
                - garch_params: Dictionary of GARCH parameters by column
                - copula_type: Type of copula to use ('t' or 'clayton')
                - copula_params: Parameters for the copula
                
        Returns:
            Self for method chaining
        """
        self.original_data = data.copy()
        self.asset_names = data.columns
        n_assets = len(self.asset_names)

        # Get parameters
        garch_params = kwargs.get('garch_params', {})
        copula_type = kwargs.get('copula_type', 't')  # Default: t-copula
        copula_params = kwargs.get('copula_params', {'df': 4})

        # Step 1: Fit GARCH models to each asset
        print(f"Fitting GARCH models to {n_assets} assets...")
        standardized_residuals = pd.DataFrame(index=data.index)

        for asset in self.asset_names:
            # Create GARCH model with parameters (if provided) or defaults
            params = garch_params.get(asset, {})
            garch_model = GARCHModel(
                omega=params.get('omega', 0.01),
                alpha=params.get('alpha', 0.1),
                beta=params.get('beta', 0.8)
            )

            # Fit GARCH model to asset returns
            returns = data[asset].values
            garch_model.fit(returns)

            # Store fitted model
            self.garch_models.append((asset, garch_model))

            # Store standardized residuals for copula modeling
            standardized_residuals[asset] = garch_model.residuals

        self.standardized_residuals = standardized_residuals

        # Step 2: Transform to uniform margins
        u_data = pd.DataFrame()
        for col in standardized_residuals.columns:
            u_data[col] = stats.rankdata(
                standardized_residuals[col]) / (len(standardized_residuals) + 1)

        # Step 3: Fit copula
        if copula_type.lower() == 't':
            self.copula = {
                'type': 't',
                'corr_matrix': u_data.corr().values,
                'df': copula_params.get('df', 4)
            }
        elif copula_type.lower() == 'clayton':
            # For simplicity, we'll use pairwise Kendall's tau to set Clayton parameters
            self.copula = {
                'type': 'clayton',
                'tau_matrix': u_data.corr(method='kendall').values
            }
        else:
            raise ValueError(f"Unsupported copula type: {copula_type}")

        # Step 4: Compute tail dependence matrix
        self.tail_dependence_matrix = self._compute_tail_dependence(u_data)

        self.is_fitted = True
        print("CoVaR model fitting completed successfully.")
        return self

    def _compute_tail_dependence(self, u_data: pd.DataFrame, threshold: float = 0.1) -> np.ndarray:
        """
        Compute empirical tail dependence matrix.
        
        Args:
            u_data: Data transformed to uniform margins
            threshold: Threshold for tail events
            
        Returns:
            Matrix of lower tail dependence coefficients
        """
        n_assets = len(u_data.columns)
        tail_dep = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    tail_dep[i, j] = 1.0
                    continue

                x = u_data.iloc[:, i].values
                y = u_data.iloc[:, j].values

                # Lower tail dependence
                cond_prob = np.mean(y <= threshold | x <=
                                    threshold) / np.mean(x <= threshold)
                tail_dep[i, j] = cond_prob

        return tail_dep

    def compute_covar(self, conditioning_asset: str, target_assets: List[str] = None,
                      alpha: float = 0.05, **kwargs) -> Dict:
        """
        Compute Conditional Value-at-Risk (CoVaR).
        
        Args:
            conditioning_asset: Asset that experiences stress
            target_assets: Assets to compute CoVaR for (if None, use all other assets)
            alpha: Confidence level
            **kwargs: Additional parameters
                - stress_level: Level of stress for conditioning asset (percentile)
                - n_simulations: Number of simulations
                
        Returns:
            Dictionary with CoVaR values
        """
        self._check_fitted()

        # Default: 5th percentile
        stress_level = kwargs.get('stress_level', 0.05)
        n_simulations = kwargs.get('n_simulations', 10000)

        # If no target assets specified, use all other assets
        if target_assets is None:
            target_assets = [
                asset for asset in self.asset_names if asset != conditioning_asset]

        # Get index of conditioning asset
        if conditioning_asset not in self.asset_names:
            raise ValueError(f"Asset {conditioning_asset} not in the model")

        cond_idx = list(self.asset_names).index(conditioning_asset)

        # Step 1: Simulate from the fitted copula
        # For simplicity, we'll use a t-copula or Clayton copula
        if self.copula['type'] == 't':
            # Simulate from t-copula
            n_assets = len(self.asset_names)
            df = self.copula['df']
            corr_matrix = self.copula['corr_matrix']

            # Generate multivariate normal samples
            mvn_samples = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=corr_matrix,
                size=n_simulations
            )

            # Generate chi-square random variable
            chi_square = np.random.chisquare(df=df, size=n_simulations) / df
            chi_square = chi_square.reshape(-1, 1)

            # Transform to t distribution
            t_samples = mvn_samples / np.sqrt(chi_square)

            # Transform to uniform using the t CDF
            u_samples = np.zeros_like(t_samples)
            for i in range(n_assets):
                u_samples[:, i] = t.cdf(t_samples[:, i], df=df)

        elif self.copula['type'] == 'clayton':
            # For Clayton, we'll use a simplified 2D approach for each pair
            # Just for demonstration purposes
            u_samples = np.random.uniform(
                size=(n_simulations, len(self.asset_names)))

        else:
            raise ValueError(f"Unsupported copula type: {self.copula['type']}")

        # Step 2: Apply GARCH volatility and generate returns
        simulated_returns = pd.DataFrame(index=range(n_simulations))

        for i, (asset, garch_model) in enumerate(self.garch_models):
            # Transform uniform samples to standard normal
            z = norm.ppf(u_samples[:, i])

            # Apply GARCH volatility (use last estimated volatility)
            vol = garch_model.conditional_volatility[-1]
            simulated_returns[asset] = garch_model.mean + vol * z

        # Step 3: Compute CoVaR
        # Find observations where conditioning asset is in stress
        cond_returns = simulated_returns[conditioning_asset].values
        stress_threshold = np.percentile(cond_returns, stress_level * 100)
        stress_mask = cond_returns <= stress_threshold

        # Compute VaR and CoVaR
        covar_dict = {}
        for target in target_assets:
            # Unconditional VaR
            target_returns = simulated_returns[target].values
            var_target = np.percentile(target_returns, alpha * 100)

            # Conditional VaR (CoVaR)
            target_returns_conditional = target_returns[stress_mask]
            covar_target = np.percentile(
                target_returns_conditional, alpha * 100)

            # ΔCoVaR
            delta_covar = covar_target - var_target

            covar_dict[target] = {
                'VaR': var_target,
                'CoVaR': covar_target,
                'DeltaCoVaR': delta_covar
            }

        return covar_dict

    def simulate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Simulate from the fitted CoVaR Copula model.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional simulation parameters
                - seed: Random seed
                - stress_scenario: Dictionary specifying stress conditions
                
        Returns:
            DataFrame with simulated asset returns
        """
        self._check_fitted()

        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        stress_scenario = kwargs.get('stress_scenario', None)

        # Step 1: Simulate from the fitted copula
        n_assets = len(self.asset_names)

        if self.copula['type'] == 't':
            # Simulate from t-copula
            df = self.copula['df']
            corr_matrix = self.copula['corr_matrix']

            # Generate multivariate normal samples
            mvn_samples = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=corr_matrix,
                size=n_samples
            )

            # Generate chi-square random variable
            chi_square = np.random.chisquare(df=df, size=n_samples) / df
            chi_square = chi_square.reshape(-1, 1)

            # Transform to t distribution
            t_samples = mvn_samples / np.sqrt(chi_square)

            # Transform to uniform using the t CDF
            u_samples = np.zeros_like(t_samples)
            for i in range(n_assets):
                u_samples[:, i] = t.cdf(t_samples[:, i], df=df)

        elif self.copula['type'] == 'clayton':
            # For simplicity, use a uniform distribution
            u_samples = np.random.uniform(size=(n_samples, n_assets))

        # Step 2: Apply GARCH volatility and generate returns
        simulated_returns = pd.DataFrame(index=range(n_samples))

        for i, (asset, garch_model) in enumerate(self.garch_models):
            # Transform uniform samples to standard normal
            z = norm.ppf(u_samples[:, i])

            # Apply GARCH volatility (use last estimated volatility)
            vol = garch_model.conditional_volatility[-1]
            simulated_returns[asset] = garch_model.mean + vol * z

        # Apply stress scenario if specified
        if stress_scenario is not None:
            for asset, percentile in stress_scenario.items():
                if asset in simulated_returns.columns:
                    idx = list(self.asset_names).index(asset)
                    threshold = np.percentile(
                        simulated_returns[asset], percentile * 100)
                    stress_mask = simulated_returns[asset] <= threshold

                    # Only keep stressed scenarios
                    simulated_returns = simulated_returns[stress_mask]

        return simulated_returns

    def compute_risk_measures(self, **kwargs) -> Dict:
        """
        Compute risk measures specific to CoVaR Copula model.
        
        Args:
            **kwargs: Parameters for risk calculation
                - alpha: Confidence level for VaR and CoVaR
                - conditioning_assets: List of assets to condition on
                - n_simulations: Number of simulations
                
        Returns:
            Dictionary of risk measures
        """
        self._check_fitted()

        alpha = kwargs.get('alpha', 0.05)
        conditioning_assets = kwargs.get(
            'conditioning_assets', [self.asset_names[0]])
        n_simulations = kwargs.get('n_simulations', 10000)

        risk_measures = {}

        # Compute CoVaR for each conditioning asset
        for cond_asset in conditioning_assets:
            covar = self.compute_covar(
                conditioning_asset=cond_asset,
                alpha=alpha,
                n_simulations=n_simulations
            )
            risk_measures[f'CoVaR_{cond_asset}'] = covar

        # Compute systemic risk contribution
        # Sum of ΔCoVaR across all target assets
        for cond_asset in conditioning_assets:
            systemic_impact = sum(
                covar['DeltaCoVaR']
                for asset, covar in risk_measures[f'CoVaR_{cond_asset}'].items()
            )
            risk_measures[f'Systemic_Impact_{cond_asset}'] = systemic_impact

        # Overall system stress scenario
        # All conditioning assets in stress simultaneously
        stress_scenario = {asset: alpha for asset in conditioning_assets}
        system_stress = self.simulate(
            n_samples=n_simulations,
            stress_scenario=stress_scenario
        )

        # Compute VaR under system stress
        stress_var = {}
        for asset in self.asset_names:
            if asset not in conditioning_assets:
                stress_var[asset] = np.percentile(
                    system_stress[asset], alpha * 100)

        risk_measures['System_Stress_VaR'] = stress_var

        return risk_measures

    def plot_tail_dependence(self) -> plt.Figure:
        """
        Plot tail dependence matrix.
        
        Returns:
            Matplotlib figure
        """
        self._check_fitted()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(self.tail_dependence_matrix, cmap='YlOrRd')

        # Add text annotations
        for i in range(len(self.asset_names)):
            for j in range(len(self.asset_names)):
                ax.text(j, i, f'{self.tail_dependence_matrix[i, j]:.2f}',
                        ha='center', va='center',
                        color='black' if self.tail_dependence_matrix[i, j] < 0.7 else 'white')

        # Add labels
        ax.set_xticks(range(len(self.asset_names)))
        ax.set_yticks(range(len(self.asset_names)))
        ax.set_xticklabels(self.asset_names, rotation=45)
        ax.set_yticklabels(self.asset_names)

        ax.set_title('Lower Tail Dependence Matrix')
        fig.colorbar(im, ax=ax, label='Tail Dependence Coefficient')

        plt.tight_layout()
        return fig
