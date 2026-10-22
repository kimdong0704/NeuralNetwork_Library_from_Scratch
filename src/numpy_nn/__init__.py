"""Vectorized neural network: every layer processes a whole batch with NumPy matrix operations."""
from .layer import Layer
from .network import Module, Network

__all__ = ["Layer", "Module", "Network"]
