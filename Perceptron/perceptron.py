import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union, Callable
from activation import Activation

class Perceptron:
    """Définis le modèle du perceptron."""
    def __init__(self, shape: int, activation_name: str):
        self.W, self.b = self.__init_neuron__(shape)
        self.A = self.__activation__(activation_name)
        self.activation_name = activation_name

    def Z(self, X: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Modèle Linéaire du perceptron."""
        return X.dot(W) + b

    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        """Réalise la propagation du vecteur X dans le neurone."""
        Z = self.Z(X, self.W, self.b)
        Y_pred = self.A(Z)
        return Y_pred

    def update(self, W: np.ndarray, b: np.ndarray):
        """Mets à jour les poids et le biais du neurone."""
        try:
            if self.W.shape != W.shape:
                raise ValueError("La forme des poids n'est pas compatible.")
            elif self.b.shape != b.shape:
                raise ValueError("La forme des biais n'est pas compatible.")
            self.W = W
            self.b = b
        except AttributeError:
            pass

    def __init_neuron__(self, shape: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initalise les poids et le biais du neurone."""
        W = np.random.random(shape)
        b = np.random.random(1)
        return W, b

    def __activation__(self, activation_name: str) -> Union[Callable, None]:
        """Définis la fonction d'activation."""
        try:
            activation_function = getattr(Activation, activation_name)
            return activation_function
        except AttributeError:
            print('Fonction d\'activation non présente dans la classe "Activation": ', activation_name)
