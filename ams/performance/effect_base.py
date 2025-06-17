# effect_base.py
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class EffectBase(ABC):
    """Shared helpers for ActionEffect & HazardEffect."""

    _FIELDS: List[str] = []          # each subclass fills this

    def __init__(self, name: str) -> None:
        self.name = name
        # every numeric field becomes a state→triangular-tuple dict
        for fld in self._FIELDS:
            setattr(self, fld, {})

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _triangular(sample: List[float]) -> float:
        a, m, b = sample
        try:
            return np.random.triangular(a, m, b)
        except ValueError:            # a==m==b, etc.
            return m

    def get(self, field: str, state: int, default):
        return self._triangular(getattr(self, field).get(state, default))

    def set(self, field: str, effect: Dict[int, List[float]]) -> None:
        getattr(self, field).update({int(k): v for k, v in effect.items()})

    # ------------------------------------------------------------------------
    # @abstractmethod
    # def generate_windows(self, ic: int) -> list["_EffectWindow"]:
    #     """Return the list of transient windows this effect triggers at *ic*."""
