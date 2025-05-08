"""
Created on 05/08/2025

@author: Aryan

Filename: GumbelCopula.py

Relative Path: src/copula/GumbelCopula.py
"""

import numpy as np

from typing import Dict

from copula.CopulaDistribution import CopulaDistribution


class GumbelCopula(CopulaDistribution):
    """Gumbel copula implementation (bivariate only)."""

    def __init__(self):
        """Initialize a Gumbel copula."""
        super().__init__(name="Gumbel")

    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:
        """
        Simulate samples from a bivariate Gumbel copula.
        
        Args:
            n_samples: Number of samples to generate
            params: Dictionary containing 'theta' parameter (must be >= 1)
            
        Returns:
            Uniform samples from the Gumbel copula
        """
        theta = params.get('theta', 1.5)  # Must be >= 1

        if theta < 1:
            raise ValueError("Theta must be >= 1 for Gumbel copula.")

        # Generate stable random variable (V) with Laplace transform exp(-t^(1/theta))
        v = self._generate_stable_variable(n_samples, theta)

        # Generate independent uniform variables
        u1 = np.random.uniform(0, 1, n_samples)
        u2 = np.random.uniform(0, 1, n_samples)

        # Transform to Gumbel copula using conditional distribution method
        t1 = -np.log(u1)
        t2 = -np.log(u2)

        # Create Gumbel copula using the Marshall-Olkin algorithm
        e1 = -np.log(u1) / v
        e2 = -np.log(u2) / v

        # Transform back to uniform
        u1_final = np.exp(-e1)
        u2_final = np.exp(-e2)

        return np.column_stack((u1_final, u2_final))

    def _generate_stable_variable(self, n_samples: int, theta: float) -> np.ndarray:
        """
        Generate stable random variable for Gumbel copula.
        
        Args:
            n_samples: Number of samples
            theta: Gumbel copula parameter
            
        Returns:
            Samples from stable distribution
        """
        # Parameter for stable distribution
        alpha = 1.0 / theta

        # Generate uniform random variables
        u = np.random.uniform(0, 1, n_samples) * np.pi - np.pi/2
        w = np.random.exponential(1, n_samples)

        # Use Chambers-Mallows-Stuck method to generate stable random variable
        gamma = np.cos(u) ** (-theta)
        s = (np.sin(theta * u) / np.sin(u)) ** theta

        return s / gamma * (np.cos(u * (1-theta)) / w) ** ((1-theta)/theta)
