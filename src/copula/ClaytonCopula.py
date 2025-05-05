"""
Created on 05/05/2025

@author: Aryan

Filename: ClaytonCopula.py

Relative Path: src/copula/ClaytonCopula.py
"""

import numpy as np

from typing import Dict

from copula.CopulaDistribution import CopulaDistribution


class ClaytonCopula(CopulaDistribution):
    """Clayton copula implementation (bivariate only)."""

    def __init__(self):
        """Initialize a Clayton copula."""
        super().__init__(name="Clayton")

    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:
        """
        Simulate samples from a bivariate Clayton copula using inverse transform.
        """
        theta = params.get('theta', 2.0)  # Must be > 0

        if theta <= 0:
            raise ValueError("Theta must be > 0 for Clayton copula.")

        # Step 1: Generate uniform U1 and independent W ~ Uniform[0, 1]
        u1 = np.random.uniform(0, 1, n_samples)
        w = np.random.uniform(0, 1, n_samples)

        # Step 2: Use inverse transform formula for Clayton
        u2 = (w**(-theta / (1 + theta)) * u1 **
              (-theta) - 1 + u1**(-theta))**(-1/theta)

        return np.column_stack((u1, u2))
