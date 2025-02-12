# Proyecto de Segmentación y Clasificación

## Segmentación

### Datasets
- **No Tráfico**: [Enlace](https://app.roboflow.com/spacelab/no-traffic-dataset/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- **DJI Video**: [Enlace](https://app.roboflow.com/spacelab/drone-trajectory/2)
- **Tráfico**: [Enlace](https://app.roboflow.com/workspace-chkqy/traffic-dataset-yoa9p/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- **VisDrone**: [Enlace](https://app.roboflow.com/ai-faeyy/object-segmentation-3zyfg/6)

### Comet
El tracking de experimentos se realiza con **Comet**. La cuenta está configurada con el correo de **IA**, y la contraseña por defecto.

🔗 **Comet Dashboard**: [Acceder](https://www.comet.com/ia-spacelab/eye-segmentation/view/new/panels)  

#### Convención de nombres en Comet:
Los modelos están etiquetados con un formato específico, por ejemplo:
```
best-InitParam_S_NoW_New_200.pt
```
Donde:
- `InitParam`: Parámetros iniciales (valores por defecto en el script de entrenamiento)
- `S`: Modelo de segmentación pequeño, ajustado con YOLOv11s
- `NoW`: No se usaron pesos predefinidos
- `200`: Número de épocas
- `NEW`: Entrenado con un nuevo dataset (+400 imágenes)

---

### Entrenamiento en la Computadora de Roberto
Pasos para entrenar el modelo:
1. Inicializar el ambiente de entrenamiento
2. Configurar la clave de API de Comet:
   ```sh
   set COMET_API_KEY=DVa5iTRzKm7M5stZqhf8eEo1T
   ```
   *(Esta clave se encuentra en el perfil de la cuenta de Comet)*
3. Ejecutar el script de entrenamiento:
   ```sh
   python model_train_eye.py
   ```

📁 **Archivos almacenados en la computadora de Roberto**  

---

### Próximos pasos:
✅ Convertir el modelo a TensorRT:
```sh
yolo export model=yolo11n.pt format=engine  # Crea 'yolo11n.engine'
```

📌 **Revisar que todo lo importante esté en el repositorio de GitHub**  

---

## Clasificación

### Dataset
- **Humos y No Humos**: [Enlace](https://huggingface.co/datasets/IzelAv/Smoke/tree/main)

### Modelos
- **Top modelos**: [Enlace](https://huggingface.co/IzelAv/alpha01/tree/main)
