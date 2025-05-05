"""
Created on 05/05/2025

@author: Aryan

Filename: FinancialCopulaModel.py

Relative Path: src/copula/TimeSeries/FinancialCopulaModel.py
"""

from abc import ABC, abstractmethod
from typing import Dict


import pandas as pd
from copula.TimeSeries.FinancialCopulaModel import FinancialCopulaModel


class FinancialCopulaModel(ABC):
    """
    Abstract base class for advanced financial copula models.
    """

    def __init__(self, name: str, description: str, use_case: str):
        """
        Initialize the financial copula model.
        
        Args:
            name: Name of the model
            description: Description of what the model captures
            use_case: Primary use case for the model
        """
        self.name = name
        self.description = description
        self.use_case = use_case
        self.is_fitted = False

    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} - {self.description}"

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'FinancialCopulaModel':
        """
        Fit the model to the given data.
        
        Args:
            data: Financial time series data
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def simulate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Simulate samples from the fitted model.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional simulation parameters
            
        Returns:
            Simulated financial data
        """
        pass

    @abstractmethod
    def compute_risk_measures(self, **kwargs) -> Dict:
        """
        Compute model-specific risk measures.
        
        Args:
            **kwargs: Parameters for risk calculation
            
        Returns:
            Dictionary of risk measures
        """
        pass

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before using this method.")
