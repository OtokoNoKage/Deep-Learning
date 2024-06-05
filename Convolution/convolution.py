import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Literal

def conv2D(X: np.ndarray, m: int) -> np.ndarray:
    n,p,c = X.shape
    kernel = np.random.random((m,m,c))
    # Taille de la feature map ()
    fm_size = ((n - m + 1), (p - m + 1))

    # feature map
    fm = np.zeros((fm_size[0], fm_size[1]))
    for i in range(fm_size[0]):
        for j in range(fm_size[1]):
            fm[i,j] = np.sum(kernel*X[i:i+m, j:j+m])

    return fm

Rhaenyra = Image.open('/data/Documents/GM/IA/GitHub/Deep-Learning/Convolution/Images/Rhaenyra.png')
Rhaenyra = np.array(Rhaenyra.resize((448,448)))
Rhaenyra = Rhaenyra / Rhaenyra.max()
conv = conv2D(Rhaenyra, 8)
fig, ax = plt.subplots()
print(f'conv.shape = {conv.shape}')
ax.imshow(conv)
plt.show()


            
