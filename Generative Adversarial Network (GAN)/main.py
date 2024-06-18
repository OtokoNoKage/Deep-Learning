import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from model import Discriminator, Generator
from dataset import Compose, Dataset
from train import train_fn, noise
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from keras.datasets.mnist import load_data

(X_train, Y_train), (X_test, Y_test) = load_data()
print(f'Nombre d\'éléments dans l\'entraînement: {len(X_train)}')
print(f'Nombre d\'éléments dans le test: {len(X_test)}')
X_train = np.float32(X_train) / X_train.max()



# Hyperparamètres Entraînement
EPOCHS = 2500
LEARNING_RATE = 0.5
MILESTONES = [50]
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 100
SIZE = 28

# Hyperparamètres Dataset/Dataloader
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True
SHUFFLE = True
DROP_LAST = True

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    # Optimiseur
    doptimizer = optim.SGD(discriminator.parameters(), lr=LEARNING_RATE, momentum=0.5)
    goptimizer = optim.SGD(generator.parameters(), lr=LEARNING_RATE, momentum=0.5)


    # Pas variant en fonction des epochs
    dscheduler = MultiStepLR(doptimizer, milestones=MILESTONES, gamma=0.1)
    gscheduler = MultiStepLR(goptimizer, milestones=MILESTONES, gamma=0.1)

    # Transformateur: transforme les images et les vecteurs label_matrix en tenseur
    transform = Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),])
    
    train_dataset = Dataset(X_train, transform)
    loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST)

    dmean_loss = []
    gmean_loss = []
    # Apprentissage
    loop = tqdm(range(1, EPOCHS+1))
    for _ in loop:
        dloss, gloss = train_fn(1, loader, generator, discriminator, goptimizer, doptimizer, device)

        dmean_loss.append(dloss)
        gmean_loss.append(gloss)

        dscheduler.step()
        gscheduler.step()

        loop.set_postfix(dloss=dloss, gloss=gloss)

    plt.plot(dmean_loss)
    plt.plot(gmean_loss)
    plt.show()

    NB_IMAGES = 25
    z = noise(NB_IMAGES, dim=100).to(device)
    x = generator(z)
    plt.figure(figsize=(17, 17))
    for i in range(NB_IMAGES):
        plt.subplot(5, 5, 1 + i)
        plt.axis('off')
        plt.imshow(x[i].data.cpu().numpy().reshape(28, 28), cmap='gray')
    plt.show()