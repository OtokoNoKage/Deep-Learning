import os
os.system('clear')

import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from perceptron import Perceptron
from loss import LogLoss, MSELoss
from train import Train
from eval import Eval
from animation import Animation

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)

PATH = dir_path + '/Plots/'
if __name__ == "__main__":
    # Dataset
    classes = 2
    X, Y = make_blobs(n_samples=1000, n_features=2, centers=classes, random_state=0)

    # Normalisation
    X[:, 0] = X[:, 0] / X[:, 0].max()
    X[:, 1] = X[:, 1] / X[:, 1].max()

    # Divise en Train et Validation
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Convertis les données numpy en jax.numpy
    X_jax_train = jnp.array(X_train)
    X_jax_validation = jnp.array(X_validation)

    Y_jax_train = jnp.array(Y_train)
    Y_jax_validation = jnp.array(Y_validation)

    # Modèle
    activation_name = "Sigmum"
    model = Perceptron(shape=X.shape[1], activation_name=activation_name)

    # Fonction coût
    Loss_Fn = MSELoss()

    # Hyperparamètres
    epochs = 1000
    lr = 1e-1

    # Listes contenant les poids et biais du modèle.
    List_W = [model.W]
    List_b = [model.b]

    # Listes contenant les pertes et précisions moyennes par epochs
    mean_train_loss = []
    mean_train_accuracy = []
    mean_val_loss = []
    mean_val_accuracy = []

    loop = tqdm(range(1, epochs+1), leave=True)
    for _ in loop:
        # Entraînement
        train_loss, train_accuracy = Train(model, X_jax_train, Y_jax_train, Loss_Fn, lr)
        mean_train_loss.append(train_loss)
        mean_train_accuracy.append(train_accuracy)

        # Évaluation
        val_loss, val_accuracy = Eval(model, X_jax_validation, Y_jax_validation, Loss_Fn)
        mean_val_loss.append(val_loss)
        mean_val_accuracy.append(val_accuracy)

        loop.set_postfix(train_loss=round(train_loss, 2), train_accuracy = train_accuracy, val_loss=round(val_loss, 2), val_accuracy=val_accuracy)

        # Sauvegarde des poids de l'itération
        List_W.append(model.W)
        List_b.append(model.b)

    # Transformation des listes en numpy.ndarray
    List_W = np.array(List_W)
    List_b = np.array(List_b).reshape((len(List_b)))

    # Crée les animations
    ani = Animation(classes, epochs, 100)
    ani.make(model, X_train, Y_train, List_W, List_b, mean_train_loss, mean_train_accuracy, lr, PATH, 'Train-Perceptron-(' + activation_name + '-' + Loss_Fn.name() + ')', Loss_Fn.name())
    ani.make(model, X_validation, Y_validation, List_W, List_b, mean_val_loss, mean_val_accuracy, lr, PATH, 'Validation-Perceptron-(' + activation_name + '-' + Loss_Fn.name() + ')', Loss_Fn.name())