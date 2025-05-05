from typing import Dict

import numpy as np


class CopulaDistribution:
    """Base class for copula distributions."""

    def __init__(self, name: str):
        """
        Initialize the copula distribution.
        
        Args:
            name: Name of the copula distribution
        """
        self.name = name

    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:
        """
        Simulate samples from the copula.
        
        Args:
            n_samples: Number of samples to generate
            params: Parameters for the copula
            
        Returns:
            Uniform samples from the copula
        """
        raise NotImplementedError("Subclasses must implement this method")
