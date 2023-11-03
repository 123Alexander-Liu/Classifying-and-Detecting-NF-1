# Classifying-and-Detecting-NF-1

## Overview
This repository contains the implementation of a deep learning model for classifying and detecting Neurofibromatosis Type 1 (NF-1) through image segmentation. The model is based on a U-Net++ architecture, suitable for both binary and multi-class segmentation tasks.

## Environment Configuration
To run the scripts in this repository, the following environment configuration is required:

- Python version: 3.8.8
- PyTorch version: 1.8.1
- Torchvision version: 0.9.1

Ensure that you have the correct versions installed to avoid any compatibility issues.

## Configuration Settings
The `config.py` file holds the configuration parameters for the model training and prediction. Key parameters include:

- `epochs`: Number of training epochs.
- `batch_size`: Size of each batch of data.
- `validation`: Percentage of data to be used for validation.
- `lr`: Learning rate for the optimizer.
- `lr_decay_milestones`: Epochs at which to decay the learning rate.
- `lr_decay_gamma`: Multiplicative factor of learning rate decay.
- `optimizer`: Type of optimizer used for training.
- `n_channels`: Number of input channels in the data.
- `n_classes`: Number of classes for segmentation.
- `scale`: Factor to scale the input images.
- `save_cp`: Boolean to decide whether to save model checkpoints.
- `model`: Type of model to use.
- `deep supervision`: Whether to use deep supervision in training.

These settings are customizable to fit the specific needs of the dataset and computational constraints.

## Training Process
The training of the model is facilitated by `train.py`, which includes:

1. Data loading and augmentation.
2. Model initialization, optimizer and loss function setup.
3. Training loop with checkpoint saving and progress logging.
4. Validation performance evaluation.

The script is designed to be flexible, allowing for different optimizers, learning rate schedules, and the use of deep supervision.

## Prediction and Color Mapping
The `predict_color.py` script is responsible for:

1. Loading a trained model.
2. Processing input images for model prediction.
3. Applying a softmax function to derive class probabilities.
4. Converting probabilities to binary masks for each class.
5. Mapping each class mask to a designated color.
6. Overlaying the colored masks on the original images and saving the output.

This script is set up for batch processing and visualizes the segmentation results with color-coded regions.

## Graphical User Interface (GUI)
The `gui.py` script provides a user-friendly graphical interface to interact with the model. It allows users to:

1. Open and display images.
2. Run the segmentation model on the loaded images.
3. Display the original, ground truth, and predicted segmentation images.
4. Calculate and display the accuracy of the predictions.

The GUI is built using PyQt5 and provides a convenient way for users to visually assess the model's performance.

---

**Note**: The detailed behavior and capabilities of the model may depend on the full content of the `losses.py` file and other specific aspects of the project not covered in the provided files. For a complete understanding, please refer to the actual code and documentation within the repository.

---
