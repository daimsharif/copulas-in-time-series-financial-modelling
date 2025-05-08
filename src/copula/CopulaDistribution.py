"""
Base class for copula distributions.

Sub‑classes **must** implement `simulate`, returning uniform pseudo‑observations
for an arbitrary (≥ 2) dimension.  Keep all heavy maths in the child class;
this file is just the common skeleton.
"""

from typing import Dict
import numpy as np


class CopulaDistribution:
    """Abstract copula distribution."""

    def __init__(self, name: str):
        self.name = name

    def simulate(self, n_samples: int, params: Dict) -> np.ndarray:  # noqa: D401
        """
        Generate observations from the copula.

        Parameters
        ----------
        n_samples : int
            Number of points to draw.
        params : Dict
            Implementation‑specific parameters – **must** include anything
            the sub‑class needs (e.g. ‘theta’, correlation matrices, df, etc.).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, d)`` array of uniforms on (0, 1).
        """
        raise NotImplementedError("Sub‑classes implement this.")
