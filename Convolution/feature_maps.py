import os
import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)

if __name__ == "__main__":
    image = Image.open(dir_path + '/Images/Rhaenyra.png')
    image = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    EfficientNetV2S = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')
    model = EfficientNetV2S.features
    model.eval()

    BATCH = 1
    SIZE = 224
    x = torch.rand((BATCH, 3, SIZE, SIZE))
    feature_maps = model(x)[0]
    feature_maps = feature_maps[0:100, ...]

    num_feature_maps = feature_maps.shape[0]
    cols = 10
    rows = num_feature_maps // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12,12))

    for i in range(rows):
        for j in range(cols):
            fm = feature_maps[i*cols + j].detach().cpu().numpy()
            axes[i,j].imshow(fm)
            axes[i,j].axis('off')

    fig.tight_layout()
    plt.show()