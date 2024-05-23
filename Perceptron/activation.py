import jax.numpy as jnp
from typing import Tuple
from math import pi

class Activation:
    """Définis les principales fonctions d'activations pour le perceptron"""
        
    def Identity(X: jnp.ndarray) -> jnp.ndarray:
        return X

    def Heaviside(X: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(X >= 0, 1.0, 0.0)
    
    def Sigmoid(X: jnp.ndarray) -> jnp.ndarray:
        return 1 / (1 + jnp.exp(-X))

    def TanH(X: jnp.ndarray) -> jnp.ndarray:
        return (jnp.exp(X) - jnp.exp(-X)) / (jnp.exp(X) + jnp.exp(-X))

    def ArcTan(X: jnp.ndarray) -> jnp.ndarray:
        return jnp.arctan(X)

    def Sigmum(X: jnp.ndarray) -> jnp.ndarray:
        return X / (1 + jnp.abs(X))
    
class Codomain:
    """Définis les domaines d'arrivés des fonctions d'activations"""
    def __init__(self):
        pass

    def Identity() -> Tuple[float, float]:
        return (-1., 1.)

    def Heaviside() -> Tuple[float, float]:
        return (-0.05, 1.05)

    def Sigmoid() -> Tuple[float, float]:
        return (-0.05, 1.05)

    def TanH() -> Tuple[float, float]:
        return (-1.05, 1.05)

    def ArcTan() -> Tuple[float, float]:
        return (-pi / 2, pi / 2)

    def Sigmum() -> Tuple[float, float]:
        return (-1.05, 1.05)