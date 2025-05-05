"""
Created on 05/05/2025

@author: Aryan

Filename: Marginal.py

Relative Path: src/copula/Marginal.py
"""

import numpy as np


from typing import Dict


class Marginal:
    """Class representing a marginal distribution."""

    def __init__(self, name: str, distribution, params: Dict):
        """
        Initialize a marginal distribution.
        
        Args:
            name: Name of the variable
            distribution: A scipy.stats distribution
            params: Parameters for the distribution
        """
        self.name = name
        self.distribution = distribution
        self.params = params

    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Apply inverse CDF transformation.
        
        Args:
            u: Uniform samples to transform
            
        Returns:
            Samples from the marginal distribution
        """
        return self.distribution.ppf(u, **self.params)
