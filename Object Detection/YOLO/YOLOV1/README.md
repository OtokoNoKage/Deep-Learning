# **Implémentation de YOLOV1 sur PyTorch**
## **Architecture**
Modified YOLOV1:
  -  Backbone: EfficientNetV2B2S
  -  YOLOV1 Head

Les résultats sont obtenus avec la méthode Non-Maximum-Suppression (NMS):
  - Seuil de fiabilité/confiance = 0.5
  - Seuil IoU entre cadres englobants prédits = 0.2 (Parcourir le reste du tri, supprimer ceux dont la valeur IoU avec le "meilleur" cadre est égale ou supérieur à un seuil définit.)

## **Résultats**
|Version                            | Train mAP (%)| Validation mAP (%)| Test mAP (%)| Epoch Checkpoint|
|:---:                              |:---:         |:---:              |:---:        |:---:            |
|Modified YOLOV1                    |14.95         |4.15               |3.53         |108              |
|Modified YOLOV1 (Data Augmentation)|21.39         |13.13              |13.13        |93               |

## **Matrice de Confusion**

### **Validation**
<p align="center">
  <figure style="display: inline-block; margin: 0 1%;">
    <img src="./Images/M_YOLOV1_Val_CM.png" alt="YOLOV1 Validation CM" width="100%" height="auto">
    <figcaption align="center">YOLOV1 Validation CM</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 0 1%;">
    <img src="./Images/M_YOLOV1_Data_Aug_Val_CM.png" alt="YOLOV1 Data Augmentation Validation CM" width="100%" height="auto">
    <figcaption align="center">YOLOV1 Data Augmentation Validation CM</figcaption>
  </figure>
</p>

### **Test**
<p align="center">
  <figure style="display: inline-block; margin: 0 1%;">
    <img src="./Images/M_YOLOV1_Test_CM.png" alt="YOLOV1 Test CM" width="100%" height="auto">
    <figcaption align="center">YOLOV1 Test CM</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 0 1%;">
    <img src="./Images/M_YOLOV1_Data_Aug_Test_CM.png" alt="YOLOV1 Data Augmentation Test CM" width="100%" height="auto">
    <figcaption align="center">YOLOV1 Data Augmentation Test CM</figcaption>
  </figure>
</p>

## **Exemple de prédiction du modèle**

