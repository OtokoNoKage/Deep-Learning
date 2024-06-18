import torch
import random as rd
from PIL import Image
from typing import List, Dict, Tuple, Optional, Callable

class Compose(object):
    """Classe pour encapsuler des transformations d'images."""
    def __init__(self, transforms: List[Callable]):
        """
        Initialise la classe Compose avec une liste de transformations.
        """
        self.transforms = transforms

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Applique séquentiellement les transformations à une image.
        """
        for t in self.transforms:
            image = t(image)
        return image
    

class Dataset(torch.utils.data.Dataset):
    """Classe Dataset pour un modèle de type GAN."""
    def __init__(self,  dataset: List[Dict], transform: Optional[Callable]):
        """
        Initialise la classe Dataset.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtient un exemplaire de la base de données à partir de son indice et de sa représentation/labellisation tensorielle par le modèle.
        """
        index = rd.randint(0, self.__len__()-1)
        image = Image.fromarray(self.dataset[index])

        if self.transform:
            image = (self.transform(image)).float()
        
        return image.flatten()