"""
Created on 05/05/2025

@author: Aryan

Filename: GARCHModel.py

Relative Path: src/copula/TimeSeries/GARCHModel.py
"""

from typing import Callable, Optional

import numpy as np




from scipy.optimize import minimize

class GARCHModel:
    """
    Simple GARCH(1,1) model implementation for volatility modeling.
    """

    def __init__(self, omega: float = 0.01, alpha: float = 0.1, beta: float = 0.8):
        """
        Initialize GARCH model parameters.
        
        Args:
            omega: Constant term in variance equation
            alpha: ARCH parameter (impact of past returns)
            beta: GARCH parameter (persistence of volatility)
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.is_fitted = False
        self.residuals = None
        self.conditional_volatility = None
        self.mean = 0

    def fit(self, returns: np.ndarray,
            method: str = 'BFGS',
            max_iter: int = 100) -> 'GARCHModel':
        """
        Fit GARCH(1,1) model to return data.
        
        Args:
            returns: Array of asset returns
            method: Optimization method
            max_iter: Maximum iterations for optimization
            
        Returns:
            Self for method chaining
        """
        # Initial parameters [omega, alpha, beta]
        initial_params = np.array([self.omega, self.alpha, self.beta])

        # Define negative log likelihood function for GARCH(1,1)
        def neg_log_likelihood(params):
            omega, alpha, beta = params

            # Check parameter constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return np.inf

            n = len(returns)
            sigma2 = np.zeros(n)  # Conditional variance

            # Initial variance is sample variance
            sigma2[0] = np.var(returns)

            # Compute variance for each time step
            for t in range(1, n):
                sigma2[t] = omega + alpha * \
                    returns[t-1]**2 + beta * sigma2[t-1]

            # Log likelihood calculation
            ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
            return -ll

        # Parameter constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        constraints = ({'type': 'ineq', 'fun': lambda x: x[0]},  # omega > 0
                       {'type': 'ineq', 'fun': lambda x: x[1]},  # alpha >= 0
                       {'type': 'ineq', 'fun': lambda x: x[2]},  # beta >= 0
                       {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]})  # alpha + beta < 1

        # Optimize
        result = minimize(neg_log_likelihood,
                          initial_params,
                          method="SLSQP",
                          constraints=constraints,
                          options={'maxiter': max_iter})

        # Update parameters
        self.omega, self.alpha, self.beta = result.x
        self.mean = np.mean(returns)

        # Compute conditional volatility and standardized residuals
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = self.omega + self.alpha * \
                returns[t-1]**2 + self.beta * sigma2[t-1]

        self.conditional_volatility = np.sqrt(sigma2)
        self.residuals = (returns - self.mean) / self.conditional_volatility
        self.is_fitted = True

        return self

    def forecast_volatility(self, n_steps: int, last_return: float = None,
                            last_var: float = None) -> np.ndarray:
        """
        Forecast volatility for future periods.
        
        Args:
            n_steps: Number of steps to forecast
            last_return: Last observed return
            last_var: Last observed variance
            
        Returns:
            Array of forecasted volatilities
        """
        if not self.is_fitted and (last_return is None or last_var is None):
            raise RuntimeError(
                "Model must be fitted or provide last return and variance")

        if last_return is None:
            last_return = self.residuals[-1] * self.conditional_volatility[-1]

        if last_var is None:
            last_var = self.conditional_volatility[-1]**2

        forecasted_var = np.zeros(n_steps)

        # First step forecast
        forecasted_var[0] = self.omega + self.alpha * \
            last_return**2 + self.beta * last_var

        # Remaining steps
        for t in range(1, n_steps):
            forecasted_var[t] = self.omega + \
                (self.alpha + self.beta) * forecasted_var[t-1]

        return np.sqrt(forecasted_var)

    def simulate(self, n_samples: int,
                 seed: Optional[int] = None,
                 burn_in: int = 100,
                 innovation_dist: Callable = None) -> np.ndarray:
        """
        Simulate returns from the GARCH process.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            burn_in: Number of initial samples to discard
            innovation_dist: Distribution for innovations (defaults to standard normal)
            
        Returns:
            Simulated return series
        """
        if seed is not None:
            np.random.seed(seed)

        if innovation_dist is None:
            def innovation_dist(size): return np.random.normal(0, 1, size)

        # Initialize arrays for variance and returns
        total_samples = n_samples + burn_in
        h = np.zeros(total_samples)
        y = np.zeros(total_samples)

        # Initial variance
        h[0] = self.omega / (1 - self.alpha - self.beta)

        # Generate process
        for t in range(1, total_samples):
            # Generate innovation
            z = innovation_dist(1)[0]

            # Update volatility
            h[t] = self.omega + self.alpha * y[t-1]**2 + self.beta * h[t-1]

            # Generate return
            y[t] = self.mean + np.sqrt(h[t]) * z

        # Discard burn-in
        return y[burn_in:]
