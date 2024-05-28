import numpy as np
import imageio.v2 as io
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from activation import Codomain

class Animation:
    def __init__(self, num_classes: int, epochs: int, frame: int) -> None:
        self.epochs = epochs
        self.frame = frame
        self.rate = int(epochs/frame) if epochs > frame else 1

    def make(self, model, X: np.ndarray, Y: np.ndarray, W: List[np.ndarray], b: List[np.ndarray], mean_loss: List[float], mean_accuracy: List[float], lr: float, path: str, file_name: str, loss_name: str) -> None:
        """
        Crée une animation afin de représenter l'évolution de la frontière de décision

        Parameters:
            X: Partie "Entraînement" de la base de données.
            Y: Partie "Entraînement" de la base de données.
            W: Liste contenant les poids de chaque itération.
            b: Liste contenant le biais de chaque itération.
            mean_loss: Liste contenant les pertes moyennes de chaque itération.
            mean_accuracy: Liste contenant les précisions moyennes de chaque itération.
            lr: Pas constant pour la méthode de descente de gradient.
            path: Chemin de sauvegarde du gif.
            file_name: Nom du fichier gif.
            loss_name: Nom de la fonction coût utilisée.

        """
        activation = model.A
        images = []
        x = np.linspace(-20, 20, X.shape[0])
        for i in tqdm(range(0, self.epochs+1, self.rate)):
            if i == 0:
                i = 1

            fig, ax = plt.subplots(2, 2, figsize=(20, 10))

            y = -(W[i][0]*x + b[i]) / W[i][1]

            # Frontière de décision
            ax[0][0].scatter(X[:, 0], X[:, 1], c=Y, cmap="rainbow")
            ax[0][0].plot(x, y, c='green', lw=3, label="Z")

            ax[0][0].set_xlim(-2,2)
            ax[0][0].set_ylim(-2,2)
            ax[0][0].axis('off')

            ax[0][0].set_title('Decision Boundary', fontsize=18)

            # Fonction d'activation
            z = X.dot(W[i]) + b[i]
            predictions = activation(z)
            ax[0][1].scatter(z, predictions, s=100, c=Y, cmap="rainbow")
            ax[0][1].spines['bottom'].set_position(('data', 0))
            ax[0][1].spines['left'].set_position(('data', 0))
            ax[0][1].spines['right'].set_visible(False)
            ax[0][1].spines['top'].set_visible(False)
            
            x = np.linspace(-3, 3, 1000)
            ax[0][1].plot(x, activation(x), c='green', lw=3, alpha=0.3)

            ax[0][1].set_xlim(-3,3)
            
            activation_domain = getattr(Codomain, model.activation_name)
            I = activation_domain()

            ax[0][1].set_ylim(I[0], I[1])

            ax[0][1].set_title('Activation Function:'+ " " + model.activation_name, fontsize=18)


            # Précision
            ax[1][0].plot(range(i),mean_accuracy[:i], color="green")
            ax[1][0].set_title('Accuracy', fontsize=18)

            # Coût (Perte) 
            ax[1][1].plot(range(i),mean_loss[:i], color="green")
            ax[1][1].set_title('Loss', fontsize=18)

            fig.suptitle(f'Epoch: {i}/{self.epochs} | Learning Rate: {lr}\nAccuracy: {round(mean_accuracy[i-1:i][0],2)} | {loss_name}', fontsize=24)

            fig.canvas.draw()
            images.append(np.array(fig.canvas.renderer.buffer_rgba()))
            plt.close(fig)

        duration = 10**-1

        io.mimsave(path + file_name + '.gif', images, duration=duration)