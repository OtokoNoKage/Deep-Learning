import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from convolution import conv2D
from pooling import MaxPool2D, AvgPool2D

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)

if __name__ == "__main__":
    image = Image.open(dir_path + '/Images/Rhaenyra.png')
    image = np.array(image)
    image = image / image.max()

    conv = conv2D(image, kernel_size=7, stride=8, padding=0)
    max = MaxPool2D(image, kernel_size=7, stride=8, padding=0)
    avg = AvgPool2D(image, kernel_size=7, stride=8, padding=0)

    fig, ax = plt.subplots(nrows=1, ncols=4)

    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title('Image')

    ax[1].imshow(conv, cmap='grey')
    ax[1].axis('off')
    ax[1].set_title('Convolution')

    ax[2].imshow(max, cmap='grey')
    ax[2].axis('off')
    ax[2].set_title('Max-Pooling')

    ax[3].imshow(avg, cmap='grey')
    ax[3].axis('off')
    ax[3].set_title('Average Pooling')

    plt.show()