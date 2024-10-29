# Training Deep Learning Models for Image Steganalysis

This project focuses on training deep learning models for image steganalysis using various advanced architectures.

## Prerequisites

- Python (version 3.10.14)
- TensorFlow, Keras
- [BossBase 0.1 S-UNIWARD 0.4 ppb](https://drive.google.com/drive/u/1/folders/1xRpbNjFOSGouPBz5UphhC5lYDn_GazMF?dmr=1&ec=wgc-drive-globalnav-goto) dataset for training
- SRM filters for feature extraction

## Usage
To start training the models, navigate to the library folder in your terminal and run:
´´´
python -m runner --epochs 400 --batch_size 8
´´´
## Architectures Included
The following architectures are available for training in this project:

- Transformer: Processes input data with multi-head attention layers.
- Capsule Network: Designed to capture spatial hierarchies in the data.
- KAN: Uses trainable activation functions with Kolmogorov-Arnold-Network for predictions.

Each model is trained with the BossBase Bow 0.1 S-UNIWARD 0.4 ppb database. SRM filters are applied for feature extraction.

### Monitoring Training with TensorBoard
After training, you can monitor the training metrics using TensorBoard. Run the following command:
´´´
tensorboard --logdir=D:\testing_by_library
´´´