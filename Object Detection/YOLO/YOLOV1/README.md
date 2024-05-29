# ***Implémentation de YOLOV1 sur PyTorch***
Modified YOLOV1:
  -  Backbone: EfficientNetV2B2s
  -  YOLOV1 Head
Les résultats sont obtenus avec la méthode Non-Maximum-Suppression (NMS):
  - Seuil de fiabilité/confiance = 0.5
  - Seuil IoU entre cadres englobants prédits = 0.2 (Parcourir le reste du tri, supprimer ceux dont la valeur IoU avec le "meilleur" cadre est égale ou supérieur à un seuil définit.)


|Version                            | Train mAP (%)| Validation mAP (%)| Test mAP (%)| Epoch Checkpoint|
|:---:                              |:---:         |:---:              |:---:        |:---:            |
|Modified YOLOV1                    |14.95         |4.15               |3.53         |108              |
|Modified YOLOV1 (Data Augmentation)|21.39         |13.13              |13.13        |93               |
