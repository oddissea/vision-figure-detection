# Localización y Clasificación de la Figura Compleja de Rey

Este repositorio contiene nuestro trabajo para el desarrollo de un sistema de visión artificial basado en deep learning para el análisis automático de la Figura Compleja de Rey, un test neuropsicológico utilizado en la evaluación del deterioro cognitivo.

## Descripción

La figura compleja de Rey es un test neuropsicológico gráfico ampliamente utilizado para la detección temprana y evaluación del deterioro cognitivo. Nuestro proyecto aborda dos tareas fundamentales mediante técnicas de aprendizaje profundo:

1. **Localización**: Detección automática de la región de interés (ROI) en imágenes escaneadas que contienen el dibujo de la figura de Rey.
2. **Clasificación/Reorientación**: Determinación de la orientación correcta de la figura para su normalización y posterior análisis.

## Estructura del Repositorio

```
.
├── data/
│   ├── REY_scan_anonim/     # Imágenes originales escaneadas
│   ├── REY_roi_rot0/        # Regiones de interés extraídas
│   ├── REY_roi_manualSelection1/ # ROIs con orientación normalizada
│   └── traza_REY.csv        # Metadatos de procesamiento
├── notebooks/
│   ├── 1_locate.ipynb       # Localización de ROI
│   └── 2_classify.ipynb     # Clasificación de orientación
├── models/
│   ├── cnn_bbox_model.pth   # Modelo CNN para localización
│   └── resnet18_orient.pth  # Modelo ResNet18 para orientación
├── src/
│   ├── datasets.py          # Clases para manejo de datasets
│   ├── models.py            # Definiciones de arquitecturas
│   ├── train.py             # Funciones de entrenamiento
│   └── utils.py             # Utilidades generales
├── README.md                # Este archivo
└── requirements.txt         # Dependencias del proyecto
```

## Metodología

### 1. Localización (Bounding Box Detection)

Para la tarea de localización, hemos implementado dos aproximaciones:

- **CNN Simple**: Arquitectura ligera con capas convolucionales seguidas de max-pooling y fully-connected para regresión de coordenadas.
- **ResNet18 Adaptada**: Modelo preentrenado con ImageNet, adaptado mediante transfer learning para la regresión de bounding boxes.

El preprocesamiento incluye:
- Redimensionamiento a 256×256 píxeles
- Normalización
- Conversión a escala de grises

### 2. Clasificación/Reorientación

Para la tarea de clasificación de orientación, implementamos:
- Enfoque de clasificación (4 clases: 0°, 90°, 180°, -90°)
- Arquitectura basada en ResNet18 con transfer learning

## Experimentos y Resultados

### Localización

| Modelo | IoU Promedio | Tiempo de Inferencia |
|--------|--------------|----------------------|
| CNN Simple | 0.8542 | 5.2 ms |
| ResNet18 | 0.9137 | 12.8 ms |

La métrica principal utilizada fue Intersection over Union (IoU), que mide la precisión de los bounding boxes predichos respecto a los reales.

### Clasificación de Orientación

| Modelo | Precisión | Recall | F1-Score |
|--------|-----------|--------|----------|
| ResNet18 | 94.7% | 93.8% | 94.2% |

## Requisitos e Instalación

```bash
# Clonar repositorio
git clone https://github.com/oddissea/rey-figure-detection.git
cd rey-figure-detection

# Crear entorno virtual
conda create -n rey-figure python=3.10
conda activate rey-figure

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Preprocesamiento de Datos

```python
from src.utils import load_and_preprocess

# Cargar y preprocesar imagen
image = load_and_preprocess("path/to/image.jpg", size=(256, 256))
```

### Localización con Modelo Entrenado

```python
import torch
from src.models import SimpleBBoxCNN

# Cargar modelo
model = SimpleBBoxCNN()
model.load_state_dict(torch.load("models/cnn_bbox_model.pth"))
model.eval()

# Predicción
with torch.no_grad():
    bbox = model(image.unsqueeze(0))
```

## Trabajo Futuro

- Implementación de segmentación semántica para identificar los 16 componentes individuales de la figura
- Desarrollo de métricas de evaluación automática basadas en la puntuación neuropsicológica estándar
- Integración de un sistema end-to-end para el análisis completo (localización, orientación y evaluación)

## Referencias

1. Rey, A. (1941). L'examen psychologique dans les cas d'encéphalopathie traumatique. Archives de Psychologie, 28, 286-340.
2. Osterrieth, P. A. (1944). Le test de copie d'une figure complexe. Archives de Psychologie, 30, 206-356.

## Autores

Fernando H. Nasser-Eddine López

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.
