
class DCCCopula(FinancialCopulaModel):
    """
    Dynamic Conditional Correlation (DCC) Copula model for capturing
    time-varying correlation structures in financial assets.
    """

    def __init__(self):
        """Initialize the DCC Copula model."""
        super().__init__(
            name="Dynamic Conditional Correlation Copula",
            description="Time-varying correlation structure",
            use_case="Dynamic joint modeling for financial assets"
        )
        self.garch_models = []
        self.dcc_params = {'a': 0.05, 'b': 0.89}  # Default DCC parameters
        self.asset_names = None
        self.standardized_residuals = None
        self.correlation_matrices = None
        self.unconditional_corr = None

    def fit(self, data: pd.DataFrame, **kwargs) -> 'DCCCopula':
        """
        Fit the DCC Copula model to financial data.
        
        Args:
            data: Financial time series returns
            **kwargs: Additional parameters for fitting
                - garch_params: Dictionary of GARCH parameters by column
                - dcc_params: Dictionary with DCC parameters ('a' and 'b')
                - method: Optimization method for DCC fitting
                
        Returns:
            Self for method chaining
        """
        self.original_data = data.copy()
        self.asset_names = data.columns
        n_assets = len(self.asset_names)
        n_obs = len(data)

        # Get parameters
        garch_params = kwargs.get('garch_params', {})
        dcc_params = kwargs.get('dcc_params', self.dcc_params)
        self.dcc_params = dcc_params

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

        # Step 2: Compute unconditional correlation matrix
        self.unconditional_corr = standardized_residuals.corr().values

        # Step 3: Implement DCC model
        a, b = dcc_params['a'], dcc_params['b']

        # Convert to numpy for faster computation
        std_resid = standardized_residuals.values

        # Initialize correlation matrices
        Q_bar = self.unconditional_corr.copy()
        Qt = np.zeros((n_obs, n_assets, n_assets))
        Rt = np.zeros((n_obs, n_assets, n_assets))

        # Initial Q is unconditional correlation
        Qt[0] = Q_bar.copy()

        # Compute Qt (quasi-correlation matrices)
        for t in range(1, n_obs):
            # Previous Q
            Q_prev = Qt[t-1]

            # Outer product of standardized residuals
            epsilon = std_resid[t-1].reshape(-1, 1)
            outer_prod = epsilon @ epsilon.T

            # Update Q
            Qt[t] = (1 - a - b) * Q_bar + a * outer_prod + b * Q_prev

        # Compute correlation matrices from Qt
        for t in range(n_obs):
            Q_diag = np.diag(np.sqrt(1 / np.diag(Qt[t])))
            Rt[t] = Q_diag @ Qt[t] @ Q_diag

        self.correlation_matrices = Rt
        self.is_fitted = True
        print("DCC model fitting completed successfully.")
        return self

    def get_correlation_at_time(self, t: int) -> np.ndarray:
        """
        Get correlation matrix at specific time index.
        
        Args:
            t: Time index
            
        Returns:
            Correlation matrix at time t
        """
        self._check_fitted()
        if t < 0 or t >= len(self.correlation_matrices):
            raise ValueError(f"Time index {t} out of bounds")

        return self.correlation_matrices[t]

    def forecast_correlation(self, horizon: int, last_correlation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast correlation matrices for future periods.
        
        Args:
            horizon: Number of steps to forecast
            last_correlation: Last observed correlation matrix (if None, use last fitted)
            
        Returns:
            Array of forecasted correlation matrices
        """
        self._check_fitted()
        a, b = self.dcc_params['a'], self.dcc_params['b']
        n_assets = len(self.asset_names)

        # Use last fitted correlation if not provided
        if last_correlation is None:
            last_correlation = self.correlation_matrices[-1]

        # Forecast correlation matrices
        forecasted_corr = np.zeros((horizon, n_assets, n_assets))
        forecasted_corr[0] = last_correlation

        for h in range(1, horizon):
            # Forecast formula for DCC
            forecasted_corr[h] = (
                1 - a - b) * self.unconditional_corr + (a + b) * forecasted_corr[h-1]

        return forecasted_corr

    def simulate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Simulate from the fitted DCC Copula model.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional simulation parameters
                - seed: Random seed
                - forecast_horizon: Number of steps ahead to forecast
                - start_idx: Starting index for simulation (default: last observation)
                
        Returns:
            DataFrame with simulated asset returns
        """
        self._check_fitted()

        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        forecast_horizon = kwargs.get('forecast_horizon', 1)
        start_idx = kwargs.get('start_idx', len(self.correlation_matrices) - 1)

        # Step 1: Forecast correlation matrices
        if start_idx < len(self.correlation_matrices) - 1:
            # Use historical correlation at start_idx
            last_corr = self.correlation_matrices[start_idx]
        else:
            # Use last historical correlation
            last_corr = self.correlation_matrices[-1]

        # Forecast correlation matrices
        forecasted_corr = self.forecast_correlation(
            forecast_horizon, last_corr)

        # Step 2: Simulate correlated standard normal variables for each forecast period
        n_assets = len(self.asset_names)
        simulated_returns = pd.DataFrame(index=range(n_samples))

        # Use the last forecasted correlation
        corr_matrix = forecasted_corr[-1]

        # Generate multivariate normal with the forecasted correlation
        z_samples = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=corr_matrix,
            size=n_samples
        )

        # Step 3: Transform to returns using GARCH volatility forecasts
        for i, (asset, garch_model) in enumerate(self.garch_models):
            # Get volatility forecast
            forecasted_vol = garch_model.forecast_volatility(forecast_horizon)

            # For multi-step ahead, use last forecasted volatility
            vol = forecasted_vol[-1]

            # Generate returns using simulated innovations and forecasted volatility
            simulated_returns[asset] = garch_model.mean + vol * z_samples[:, i]

        return simulated_returns

    def compute_risk_measures(self, **kwargs) -> Dict:
        """
        Compute risk measures specific to DCC Copula model.
        
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

        # Dynamic correlation risk measures
        # Calculate VaR based on different correlation regimes

        # Get recent correlation matrices
        recent_corr = self.correlation_matrices[-10:]

        # Find high and low correlation regimes
        avg_corr_values = []
        for corr_mat in recent_corr:
            # Calculate average off-diagonal correlation
            n = corr_mat.shape[0]
            off_diag_sum = np.sum(corr_mat) - np.sum(np.diag(corr_mat))
            avg_corr = off_diag_sum / (n * (n - 1))
            avg_corr_values.append(avg_corr)

        # Risk measures under high correlation scenario
        high_corr_idx = np.argmax(avg_corr_values)
        high_corr_matrix = recent_corr[high_corr_idx]

        # Simulate with high correlation
        high_corr_returns = self.simulate(
            n_samples=n_simulations,
            forecast_horizon=forecast_horizon,
            override_corr=high_corr_matrix
        )

        # Equal-weighted portfolio returns under high correlation
        weights = np.ones(len(self.asset_names)) / len(self.asset_names)
        high_corr_portfolio = high_corr_returns.dot(weights)

        risk_measures['High_Corr_VaR'] = np.percentile(
            high_corr_portfolio, alpha * 100)
        risk_measures['High_Corr_CVaR'] = high_corr_portfolio[high_corr_portfolio <=
                                                              risk_measures['High_Corr_VaR']].mean()

        return risk_measures

    def plot_correlation_dynamics(self) -> plt.Figure:
        """
        Plot the dynamic correlation structure over time.
        
        Returns:
            Matplotlib figure with correlation heatmaps
        """
        self._check_fitted()
        n_assets = len(self.asset_names)
        n_plots = min(4, len(self.correlation_matrices) //
                      (len(self.correlation_matrices) // 4))

        # Select time points for visualization
        indices = np.linspace(
            0, len(self.correlation_matrices) - 1, n_plots, dtype=int)

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            corr_mat = self.correlation_matrices[idx]
            im = axes[i].imshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
            axes[i].set_title(f'Correlation at t={idx}')
            for a in range(n_assets):
                for b in range(n_assets):
                    axes[i].text(b, a, f'{corr_mat[a, b]:.2f}',
                                 ha='center', va='center',
                                 color='white' if abs(
                                     corr_mat[a, b]) > 0.5 else 'black',
                                 fontsize=8)

            axes[i].set_xticks(range(n_assets))
            axes[i].set_yticks(range(n_assets))
            axes[i].set_xticklabels(self.asset_names, rotation=45)
            axes[i].set_yticklabels(self.asset_names)

        fig.colorbar(im, ax=axes, shrink=0.8, label='Correlation')
        plt.tight_layout()
        return fig
