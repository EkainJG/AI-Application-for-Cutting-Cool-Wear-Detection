# AI Application for Cutting Cool Wear Detection

AI Application for Cutting Cool Wear Detection is a Python project for training and using a U-Net model to perform binary semantic segmentation of cutting tool wear. It includes a training script for building the model and an inference script for estimating wear on new images and drawing measurement lines from the predicted mask.

## Features

- Binary semantic segmentation of wear regions
- U-Net model with an ImageNet-pretrained EfficientNet-B0 encoder
- Two-stage training strategy:
  - frozen encoder training
  - full fine-tuning
- Inference pipeline for new images
- Mask post-processing and RANSAC-based line fitting for wear measurement

## Repository structure

- `UNetTrain.py` — training pipeline for the U-Net segmentation model
- `UnetVbFN.py` — inference and measurement utilities
- `Data/` — dataset and related project data
- `best_two_TFG2.pth` — pre-trained model checkpoint

## Requirements

This project uses Python and the following main libraries:

- `torch`
- `segmentation_models_pytorch`
- `albumentations`
- `opencv-python`
- `numpy`
- `scikit-learn`

## Data preparation

The training script expects image and mask folders that are aligned by filename order. According to the code comments:

- Images should be RGB
- Masks should be grayscale
- White/255 pixels represent wear
- Black/0 pixels represent background

The current training script uses Windows-style example paths such as:

- `C:\Train_Img`
- `C:\Train_Msk`
- `C:\Val_Img`
- `C:\Val_Msk`

Update these paths before training.

## Usage

### Training

Edit the dataset paths in `UNetTrain.py`, then run:

```bash
python UNetTrain.py
```

### Inference

Edit the input image path in `UnetVbFN.py`, then run:

```bash
python UnetVbFN.py
```

## Notes

- The repository currently does not include a `requirements.txt`, so you may need to install dependencies manually.
- The scripts are written for a local Python environment and use absolute Windows paths that should be adapted to your machine.

## Suggested installation

```bash
pip install torch segmentation-models-pytorch albumentations opencv-python scikit-learn numpy
```

## License

No license file is currently included in the repository.