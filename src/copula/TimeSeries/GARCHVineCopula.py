"""
Created on 05/05/2025

@author: Aryan

Filename: GARCHVineCopula.py

Relative Path: src/copula/TimeSeries/GARCHVineCopula.py
"""

from typing import Dict
from matplotlib import pyplot as plt
from matplotlib.pylab import norm
import numpy as np
import pandas as pd
from copula.TimeSeries.FinancialCopulaModel import FinancialCopulaModel
from copula.TimeSeries.GARCHModel import GARCHModel
from copula.TimeSeries.VineCopula import VineCopula


class GARCHVineCopula(FinancialCopulaModel):
    """
    GARCH-based Vine Copula model for capturing volatility and complex dependency
    structures in high-dimensional asset modeling.
    """

    def __init__(self):
        """Initialize the GARCH-Vine Copula model."""
        super().__init__(
            name="GARCH-based Vine Copula",
            description="Volatility + complex dependency",
            use_case="High-dimensional asset modeling"
        )
        self.garch_models = []
        self.vine_copula = None
        self.asset_names = None
        self.standardized_residuals = None
        self.original_data = None

    def fit(self, data: pd.DataFrame, **kwargs) -> 'GARCHVineCopula':
        """
        Fit the GARCH-Vine Copula model to financial data.
        
        Args:
            data: Financial time series returns
            **kwargs: Additional parameters for fitting
                - garch_params: Dictionary of GARCH parameters by column
                - copula_families: List of copula families for the vine
                
        Returns:
            Self for method chaining
        """
        self.original_data = data.copy()
        self.asset_names = data.columns
        n_assets = len(self.asset_names)

        # Get GARCH parameters if provided
        garch_params = kwargs.get('garch_params', {})

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

        # Step 2: Fit vine copula to standardized residuals
        print("Fitting Vine Copula to standardized residuals...")
        copula_families = kwargs.get('copula_families', None)
        self.vine_copula = VineCopula(n_assets, copula_families)
        self.vine_copula.fit(standardized_residuals)

        self.is_fitted = True
        print("Model fitting completed successfully.")
        return self

    def simulate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Simulate from the fitted GARCH-Vine Copula model.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional simulation parameters
                - seed: Random seed
                - forecast_horizon: Number of steps ahead to forecast
                - innovation_dist: Distribution for innovations
                
        Returns:
            DataFrame with simulated asset returns
        """
        self._check_fitted()

        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        forecast_horizon = kwargs.get('forecast_horizon', 1)
        innovation_dist = kwargs.get('innovation_dist', None)

        # Step 1: Simulate from vine copula to get correlated uniform variables
        u_samples = self.vine_copula.simulate(n_samples)

        # Step 2: Transform uniforms to standardized residuals
        z_samples = pd.DataFrame(index=range(n_samples))
        for i, asset in enumerate(self.asset_names):
            z_samples[asset] = norm.ppf(u_samples[f'U{i+1}'])

        # Step 3: Simulate GARCH processes with the correlated innovations
        simulated_returns = pd.DataFrame(index=range(n_samples))

        for asset_idx, (asset, garch_model) in enumerate(self.garch_models):
            # Get volatility forecast
            forecasted_vol = garch_model.forecast_volatility(forecast_horizon)

            # For multi-step ahead, use last forecasted volatility
            vol = forecasted_vol[-1]

            # Generate returns using simulated innovations and forecasted volatility
            asset_innovations = z_samples[asset].values
            simulated_returns[asset] = garch_model.mean + \
                vol * asset_innovations

        return simulated_returns

    def compute_risk_measures(self, **kwargs) -> Dict:
        """
        Compute risk measures specific to GARCH-Vine Copula model.
        
        Args:
            **kwargs: Parameters for risk calculation
                - alpha: Confidence level for VaR and CVaR
                - forecast_horizon: Horizon for risk measures
                - n_simulations: Number of simulations for risk calculation
                
        Returns:
            Dictionary of risk measures
        """
        self._check_fitted()

        alpha = kwargs.get('alpha', 0.05)
        forecast_horizon = kwargs.get('forecast_horizon', 1)
        n_simulations = kwargs.get('n_simulations', 10000)

        # Simulate scenarios
        simulated_returns = self.simulate(
            n_samples=n_simulations,
            forecast_horizon=forecast_horizon
        )

        # Calculate risk measures
        risk_measures = {}

        # Value-at-Risk for each asset
        var_dict = {}
        for asset in self.asset_names:
            var_dict[asset] = np.percentile(
                simulated_returns[asset], alpha * 100)
        risk_measures['VaR'] = var_dict

        # Conditional Value-at-Risk (Expected Shortfall)
        cvar_dict = {}
        for asset in self.asset_names:
            returns = simulated_returns[asset].values
            var_value = var_dict[asset]
            cvar_dict[asset] = returns[returns <= var_value].mean()
        risk_measures['CVaR'] = cvar_dict

        # Portfolio VaR (assuming equal weights)
        n_assets = len(self.asset_names)
        weights = np.ones(n_assets) / n_assets
        portfolio_returns = simulated_returns.dot(weights)

        risk_measures['Portfolio_VaR'] = np.percentile(
            portfolio_returns, alpha * 100)
        risk_measures['Portfolio_CVaR'] = portfolio_returns[portfolio_returns <=
                                                            risk_measures['Portfolio_VaR']].mean()

        return risk_measures

    def plot_volatility_surface(self, **kwargs) -> plt.Figure:
        """
        Plot volatility surface for the assets.
        
        Args:
            **kwargs: Parameters for plotting
                - forecast_horizon: Number of steps to forecast
                - n_simulations: Number of simulations
                
        Returns:
            Matplotlib figure object
        """
        self._check_fitted()

        forecast_horizon = kwargs.get('forecast_horizon', 20)
        fig, axs = plt.subplots(len(self.garch_models),
                                1, figsize=(10, 3 * len(self.garch_models)))

        if len(self.garch_models) == 1:
            axs = [axs]

        for i, (asset, garch_model) in enumerate(self.garch_models):
            # Get historical volatility
            hist_vol = garch_model.conditional_volatility

            # Forecast volatility
            forecast_vol = garch_model.forecast_volatility(forecast_horizon)

            # Plot
            ax = axs[i]
            ax.plot(range(len(hist_vol)), hist_vol, label='Historical')
            ax.plot(range(len(hist_vol), len(hist_vol) + forecast_horizon), forecast_vol,
                    label='Forecast', linestyle='--')
            ax.set_title(f'Volatility for {asset}')
            ax.set_ylabel('Volatility')
            ax.set_xlabel('Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
