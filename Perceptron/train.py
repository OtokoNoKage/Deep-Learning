import jax
import jax.numpy as jnp
from typing import Tuple
from perceptron import Perceptron

def Train(model: Perceptron, X: jnp.ndarray, Y: jnp.ndarray, Loss_Fn, lr: float) -> Tuple[float, float]:
    """
    Entraîne le modèle.
    
    Returns:
        La perte et la précision du modèle sur le dataset (X, Y).
    """
    # Prédiction du modèle
    Y_pred = model.forward(X)
    
    # Calcul du coût et des gradients
    def training(W, b):
        Z = model.Z(X, W, b)
        Y_pred = model.A(Z)
        loss = Loss_Fn.forward(Y, Y_pred)
        return loss, Y_pred
    
    (loss, Y_pred), grads = jax.value_and_grad(training, argnums=(0, 1), has_aux=True)(model.W, model.b)
    
    grad_W, grad_b = grads

    # Mise à jour des poids et des biais
    W = model.W - lr * grad_W
    b = model.b - lr * grad_b
    model.update(W, b)

    # Calcul de la précision
    frontier = 0.5 if model.activation_name == "Sigmoid" else 0.
    Y_pred = Y_pred >= frontier
    Y = Y == 1
    accuracy = float(sum(Y == Y_pred) / len(Y))

    return float(loss), accuracy
