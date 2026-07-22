# Detección y medición automática del desgaste en herramientas de brochado mediante Deep Learning

Trabajo de Fin de Grado (TFG) en Ingeniería en Tecnología Industrial, desarrollado en colaboración con el **CFAA — Centro de Fabricación Avanzada Aeronáutica** (Zamudio, Bizkaia).

El proyecto aborda la **detección y cuantificación automática del desgaste de flanco** en herramientas de brochado a partir de imágenes, sustituyendo la medición manual —lenta y subjetiva— por un flujo basado en **visión por computador** y **segmentación semántica con redes neuronales**. La medición del desgaste se realiza siguiendo los criterios de la norma **ISO 3685** (anchura de la banda de desgaste, *VB*).

El sistema se divide en dos etapas:

1. **Segmentación** de la zona de desgaste con una red **U-Net** (encoder EfficientNet-B0 preentrenado en ImageNet).
2. **Medición geométrica** del desgaste sobre la máscara predicha mediante ajuste robusto de la línea de referencia con **RANSAC** y proyección de los puntos de la máscara.

---

## Estructura del repositorio

```
.
├── UNetTrain.py                                   # Entrenamiento del modelo de segmentación
├── Unet_Resnet34_hough_finalRANSAC_FN_copy_2.py   # Inferencia + medición del desgaste (ISO 3685)
├── README.md
└── data/                                          # Dataset (no incluido / gestionado localmente)
    ├── Train_Img/     # Imágenes de entrenamiento (RGB)
    ├── Train_Msk/       # Máscaras de entrenamiento (escala de grises)
    ├── Val_Img/         # Imágenes de validación (RGB)
    └── Val_Msk/         # Máscaras de validación (escala de grises)
```
---

## Dataset

- **Imágenes:** formato RGB (`.jpg`).
- **Máscaras:** escala de grises, donde **255 = zona de desgaste** y **0 = fondo**.
- Cada imagen debe tener su máscara correspondiente, **ordenadas de forma que se emparejen** al listarlas alfabéticamente (p. ej. `img_001.jpg` ↔ `msk_001.png`).
- Partición empleada: **120 imágenes de entrenamiento / 24 de validación** (144 en total).

En el `Dataset` de entrenamiento, las máscaras se binarizan con umbral (`mask > 127`) y se aplica una lógica de *ensure_positive* que reintenta el recorte aleatorio para evitar un exceso de parches vacíos (sin desgaste).

---

## Requisitos

- Python 3.9+
- GPU NVIDIA con CUDA (recomendado para el entrenamiento)

### Dependencias principales

```
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python
numpy
scikit-learn
```
---

## Uso

### 1. Entrenamiento — `UNetTrain.py`

Antes de ejecutar, **ajusta las rutas** al inicio del script (actualmente apuntan a rutas locales de Windows):

```python
train_images = ".../Train_Img"
train_masks  = ".../Train_Msk"
val_images   = ".../Val_Img"
val_masks    = ".../Val_Msk"
```

Ejecución:

```bash
python UNetTrain.py
```

#### Estrategia de entrenamiento en dos fases

El entrenamiento sigue un esquema de *transfer learning* en dos fases:

- **Fase 1 — Encoder congelado.** Solo se actualizan el *decoder* y la *segmentation head*. El encoder conserva intactas sus *features* de ImageNet, lo que permite una convergencia rápida y estable.
- **Fase 2 — Fine-tuning completo.** Se descongela toda la red y se entrena con *learning rates* diferenciados: uno muy bajo para el encoder (para no degradar el conocimiento de ImageNet) y otros mayores para el decoder y la cabeza de segmentación.

Se aplica *early stopping* sobre la métrica **Dice** de validación y un *scheduler* `ReduceLROnPlateau`.

#### Función de pérdida y métricas

- **Pérdida:** `DiceLoss` + `SoftBCEWithLogitsLoss`.
- **Métricas de validación:** Dice (F1), IoU, *accuracy* y *precision*.

#### Data augmentation

Con **Albumentations**: recorte aleatorio 512×512, *flips* horizontal/vertical, rotaciones de 90°, y variaciones fotométricas (brillo/contraste, gamma, CLAHE, ruido gaussiano, desenfoque, *sharpen*, *hue/saturation*).

#### Salidas del entrenamiento

| Fichero | Contenido |
|---|---|
| `best_model.pth` | Pesos del **mejor modelo** (mayor Dice en validación) |
| `final_model.pth` | Pesos del modelo al **finalizar** el entrenamiento |
| `Train.csv` | Registro por época (Dice, IoU, accuracy, precision, pérdidas) |
| `best_metrics.csv` | Métricas finales del mejor checkpoint |

### 2. Inferencia y medición — `UnetVbFN.py`

Ajusta la ruta de la imagen de prueba y la del checkpoint entrenado:

```python
image_path = ".../images1.jpg"
model.load_state_dict(torch.load(".../best_model.pth", map_location="cpu"))
```

Ejecución:

```bash
python UnetVbFN.py
```

#### Pipeline de medición

1. **Segmentación** de la imagen con la U-Net y binarización de la máscara predicha.
2. Selección del **componente conexo de mayor área** (descarta ruido de segmentación).
3. Extracción del **borde inferior** de la máscara mediante detección de bordes (Canny).
4. Ajuste de la **línea de referencia** al borde inferior con **RANSAC** (robusto frente a *outliers*).
5. Proyección de los puntos de la máscara sobre las direcciones **normal** y **tangente** a la línea, y cálculo de la altura de desgaste por *bins* a lo largo de la tangente.
6. **Filtrado de outliers** de las alturas mediante el criterio **IQR** (rango intercuartílico).
7. Cálculo de las medidas de desgaste según **ISO 3685** y detección de **muescas/roturas** (*chipping*) por umbral.

#### Calibración píxel → mm

La conversión de píxeles a milímetros se controla con el parámetro `pixelratio` (por defecto `1/128.4758`, es decir, 128,4758 px/mm). **Debe recalibrarse** según la resolución y el montaje óptico de cada campaña de captura.

#### Resultados que se calculan

- **VBmax**: altura máxima de desgaste (px y mm).
- **VBavg**: altura media de desgaste (px y mm).
- **Área** de la zona de desgaste (px² y mm²).
- **Área** y **número** de muescas/roturas (*chipping*).
- Imagen *overlay* con la máscara (verde), las muescas (naranja), la línea RANSAC y las líneas de medición, junto a un cuadro de texto con los valores.

---

## Modelo

- **Arquitectura:** U-Net (`segmentation_models_pytorch`).
- **Encoder:** EfficientNet-B0, pesos preentrenados en ImageNet.
- **Salida:** máscara binaria de 1 canal (1 = desgaste, 0 = fondo).

---

## Reproducibilidad

El entrenamiento fija una semilla (`SEED = 42`) para `random`, `numpy` y `torch`, y activa el modo determinista de cuDNN, de modo que las ejecuciones sean reproducibles.

---

## Notas y limitaciones

- Las **rutas están codificadas** en los scripts (rutas absolutas de Windows). Conviene parametrizarlas (argumentos de línea de comandos o fichero de configuración) antes de compartir el repositorio.
- El `pixelratio` es **específico del montaje de captura**; una calibración incorrecta invalida las medidas en mm.
- El dataset no se incluye en el repositorio y debe colocarse en la estructura de carpetas descrita.

---

## Autoría

TFG desarrollado en el marco del Grado en Ingeniería en Tecnología Industrial, en colaboración con el **CFAA (Centro de Fabricación Avanzada Aeronáutica)**, Zamudio (Bizkaia).
