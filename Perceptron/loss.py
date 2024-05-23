import numpy as np
import jax.numpy as jnp

class LogLoss:
    """Définis la fonction coût LogLoss."""
    def __init__(self):
        pass

    def forward(self, Y: jnp.ndarray, Y_pred: jnp.ndarray) -> float:
        """
        Calcul la perte entre Y et Y_pred.
        """
        # Nombre d'élements de notre dataset
        n = Y.shape[0]
        epsilon = 1e-9
        return (- 1 / n) * jnp.sum((Y.dot(jnp.log(Y_pred + epsilon)) + ((1 - Y).T).dot(jnp.log(1 - Y_pred + epsilon))))

    def name(self):
        return "LogLoss"
    
class MSELoss:
    """Définis la fonction coût MeanSquaredError."""
    def __init__(self):
        pass

    def forward(self, Y: jnp.ndarray, Y_pred: jnp.ndarray) -> float:
        n = Y.shape[0]
        return  jnp.mean(jnp.square(Y_pred - Y))
    
    def name(self):
        return "MSELoss"
