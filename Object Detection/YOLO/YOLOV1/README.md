# **Implémentation de YOLOV1 sur PyTorch**
## **Architecture**
Modified YOLOV1:
  -  Backbone: EfficientNetV2B2S
  -  YOLOV1 Head

## **Résultats**
Les résultats sont obtenus avec la méthode Non-Maximum-Suppression (NMS):
  - Seuil de fiabilité/confiance = 0.5
  - Seuil IoU entre cadres englobants prédits = 0.2 (Parcourir le reste du tri, supprimer ceux dont la valeur IoU avec le "meilleur" cadre est égale ou supérieur à un seuil définit.)

|Version                               | Train mAP (%)| Validation mAP (%)| Test mAP (%)| Epoch Checkpoint|
|:---:                                 |:---:         |:---:              |:---:        |:---:            |
|Modified YOLOV1                       |14.95         |4.15               |3.53         |108              |
|Modified YOLOV1 DA (Data Augmentation)|21.39         |13.13              |13.13        |93               |

## **Matrice de Confusion**

### **Validation**
<table>
  <tr>
    <td align="center">
      <figure>
        <img src="./Images/M_YOLOV1_Val_CM.png" width="100%">
        <figcaption style="font-family: Arial, sans-serif; font-size: 2px; font-weight: bold;">M_YOLOV1 Validation</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="./Images/M_YOLOV1_Data_Aug_Val_CM.png" width="100%">
        <figcaption style="font-family: Arial, sans-serif; font-size: 2px; font-weight: bold;">M_YOLOV1 DA Validation</figcaption>
      </figure>
    </td>
  </tr>
</table>

### **Test**
<table>
  <tr>
    <td align="center">
      <figure>
        <img src="./Images/M_YOLOV1_Test_CM.png" width="100%">
        <figcaption style="font-family: Arial, sans-serif; font-size: 2px; font-weight: bold;">M_YOLOV1 Test</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="./Images/M_YOLOV1_Data_Aug_Test_CM.png" width="100%">
        <figcaption style="font-family: Arial, sans-serif; font-size: 2px; font-weight: bold;">M_YOLOV1 DA Test</figcaption>
      </figure>
    </td>
  </tr>
</table>

## **Statistiques sur la base de données (VOC2007)**
