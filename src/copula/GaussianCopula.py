"""
Created on 05/05/2025

@author: Aryan

Filename: gaussianCopula.py

Relative Path: src/copula/gaussianCopula.py
"""

from typing import Dict

from matplotlib.pylab import norm
import numpy as np
from copula.CopulaDistribution import CopulaDistribution


class GaussianCopula(CopulaDistribution):
    """Gaussian copula implementation."""

    def __init__(self):
        """Initialize a Gaussian copula."""
        super().__init__(name="Gaussian")

    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:
        """
        Simulate samples from a Gaussian copula.
        
        Args:
            n_samples: Number of samples to generate
            params: Dictionary containing 'corr_matrix' for the correlation matrix
            
        Returns:
            Uniform samples from the Gaussian copula
        """
        corr_matrix = params.get('corr_matrix')
        dim = corr_matrix.shape[0]

        # Generate multivariate normal samples
        mvn_samples = np.random.multivariate_normal(
            mean=np.zeros(dim),
            cov=corr_matrix,
            size=n_samples
        )

        # Transform to uniform using the standard normal CDF
        uniform_samples = norm.cdf(mvn_samples)

        return uniform_samples
