import jax
import jax.numpy as jnp
from typing import Tuple
from perceptron import Perceptron

def Eval(model: Perceptron, X: jnp.ndarray, Y: jnp.ndarray, Loss_Fn) -> Tuple[float, float]:
    """
    Évalue le modèle.
    
    Returns:
        La perte et la précision du modèle sur le dataset (X, Y).
    """
    # Prédiction du modèle
    Y_pred = model.forward(X)
    
    # Calcul du coût
    loss = float(Loss_Fn.forward(Y, Y_pred))

    # Calcul de la précision
    frontier = 0.5 if model.activation_name == "Sigmoid" else 0.
    Y_pred = Y_pred >= frontier
    Y = Y == 1
    accuracy = float(sum(Y == Y_pred) / len(Y))

    return loss, accuracy
