import torch.nn as nn

# (caractéristiques, neurones)
"""generator_architecture = [(100, 128),
                          'ReLU',
                          (128, 256),
                          'ReLU',
                          (256,512),
                          'ReLU',
                          (512, 1024),
                          'ReLU',
                          (1024, 28*28),
                          'Tanh']"""

generator_architecture = [(100, 1200),
                          'ReLU',
                          (1200, 1200),
                          'ReLU',
                          (1200, 28*28),
                          'Tanh']

class Generator(nn.Module):
    """
    Générateur d'images.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.network = self._create_network(generator_architecture)

    def forward(self, noise):
        return self.network(noise)
    
    def _create_network(self, architecture):
        layers = []
        for element in architecture:
            if type(element) == tuple:
                layers.append(nn.Linear(element[0], element[1]))

            elif type(element) == str:
                try:
                    activation_function = getattr(nn, element)
                    layers.append(activation_function())
                except AttributeError:
                    print(f'Fonction d\'activation non présente dans la classe {nn}: ', element)
        return nn.Sequential(*layers)
    
"""# (caractéristiques, neurones)
discriminator_architecture = [(28*28, 128),
                          'LeakyReLU',
                          (128, 256),
                          'LeakyReLU',
                          (256,512),
                          'LeakyReLU',
                          (512, 512),
                          'LeakyReLU',
                          (512, 256),
                          'LeakyReLU',
                          (256, 128),
                          'LeakyReLU',
                          (128, 1),
                          'Sigmoid']"""

# (caractéristiques, neurones)
discriminator_architecture = [(28*28, 240),
                          'LeakyReLU',
                          (240, 240),
                          'LeakyReLU',
                          (240, 1),  
                          'Sigmoid']

class Discriminator(nn.Module):
    """
    Discriminateur d'images.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = self._create_network(discriminator_architecture)

    def forward(self, x):
        return self.network(x)
    
    def _create_network(self, architecture):
        layers = []
        for element in architecture:
            if type(element) == tuple:
                layers.append(nn.Linear(element[0], element[1]))

            elif type(element) == str:
                try:
                    activation_function = getattr(nn, element)
                    layers.append(activation_function())
                except AttributeError:
                    print(f'Fonction d\'activation non présente dans la classe {nn}: ', element)
        return nn.Sequential(*layers)