"""
Spectral modeling package.

Exports
-------
SpecX : Main model (formerly ABEDEncoder)
"""
from .model_specx import SpecX
from .model_baseline import*

__all__ = ["SpecX"]
