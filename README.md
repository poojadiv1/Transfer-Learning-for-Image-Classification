# VisionAdapt

## Transfer Learning and Fine-Tuning for Image Classification

VisionAdapt is a deep learning project that investigates the practical use of transfer learning for image classification by adapting a pretrained ResNet50 model to a binary image classification task. The project focuses on two commonly used transfer learning strategies: feature extraction and selective fine-tuning.

Rather than training a convolutional neural network entirely from randomly initialized weights, the project leverages visual representations learned by ResNet50 from the ImageNet dataset and adapts those representations to the target classification problem. The two approaches are evaluated under a controlled experimental setup to understand the trade-off between predictive performance, training time, and computational requirements.

The project is designed as a practical demonstration of how pretrained computer vision models can be incorporated into machine learning workflows where training a large deep neural network from scratch would be unnecessarily expensive or inefficient.

---

## Project Overview

Modern computer vision systems frequently rely on pretrained deep learning architectures rather than developing every model from scratch. Large convolutional neural networks trained on datasets such as ImageNet learn general visual representations that can often be transferred to new image classification problems.

VisionAdapt studies this process using ResNet50 as the pretrained backbone. The original ImageNet classification head is removed and replaced with a task-specific binary classification head designed to distinguish between cats and dogs.

The project implements and compares two approaches:

1. **Feature Extraction** — the pretrained ResNet50 convolutional base is frozen and only the newly added classification layers are trained.
2. **Fine-Tuning** — the majority of the pretrained network remains frozen while the final 30 ResNet50 layers are unfrozen and adapted to the target dataset using a significantly smaller learning rate.

The resulting models are evaluated using validation accuracy, test accuracy, training time, precision, recall, F1-score, and confusion-matrix analysis.

---

## Problem Statement

Training a deep convolutional neural network from scratch requires substantial computational resources, large amounts of labelled data, and significantly longer training cycles. For many practical image classification applications, this approach is unnecessary because pretrained networks already contain useful low-level and high-level visual representations.

The objective of this project is to determine how effectively a pretrained ResNet50 model can be adapted to a new binary image classification task and to compare the practical differences between using the pretrained network purely as a fixed feature extractor and selectively fine-tuning its deeper layers.

The experiment therefore addresses the following question:

> How does selective fine-tuning of a pretrained ResNet50 model affect classification performance and training efficiency compared with using the same model as a fixed feature extractor?

---

## Objectives

The primary objective of VisionAdapt is to implement a practical transfer-learning workflow using a pretrained ResNet50 architecture.

The project specifically aims to:

- Apply a pretrained ResNet50 model with ImageNet weights to a new binary image classification problem without retraining the entire network from scratch.
- Implement feature extraction by freezing the complete pretrained convolutional base and training a newly designed classification head.
- Implement selective fine-tuning by unfreezing only the final 30 layers of ResNet50 while keeping the earlier learned representations frozen.
- Compare the predictive performance of feature extraction and fine-tuning using a common validation and testing framework.
- Measure training time for both approaches to understand the computational cost associated with adapting pretrained models.
- Evaluate the final model using accuracy, precision, recall, F1-score, and confusion-matrix analysis.
- Demonstrate an industry-relevant workflow for adapting pretrained computer vision architectures to specialized classification tasks.
- Save the resulting trained model and experimental results in a reproducible project structure.

---

## Dataset

The project uses the Dogs vs Cats image classification dataset containing two target classes:

- Cat
- Dog

The downloaded dataset used in this implementation contains **24,989 images**. The dataset is divided into training, validation, and testing subsets using an approximately 80/10/10 split.

| Dataset Split | Images | Proportion |
|---|---:|---:|
| Training | 19,991 | ~80% |
| Validation | 2,499 | ~10% |
| Testing | 2,499 | ~10% |
| Total | 24,989 | 100% |

The dataset is not stored inside this repository because committing thousands of image files would unnecessarily increase repository size and reduce portability. The notebook contains the dataset acquisition and preparation procedure required to reproduce the experiment.

The dataset is organized into class-specific directories so that it can be loaded directly using Keras image data generators.

---

## Data Preprocessing

All images are resized to a fixed spatial resolution of **224 × 224 pixels**, matching the input dimensions expected by the ResNet50 architecture.

Pixel values are rescaled from their original integer representation to the range `[0, 1]`.

Training data uses lightweight augmentation to improve generalization and reduce sensitivity to variations in image orientation and scale. The applied augmentation operations include horizontal flipping and zoom transformations.

Validation and test images are not augmented. They are only resized and rescaled so that evaluation reflects the model's performance on the original distribution of unseen images rather than artificially modified samples.

The preprocessing pipeline therefore consists of:

- Image resizing to 224 × 224 pixels.
- Pixel-value normalization through rescaling by `1/255`.
- Horizontal flipping for training images.
- Zoom augmentation for training images.
- No augmentation during validation and testing.
- Binary class encoding for the Cat and Dog categories.

---

## Model Architecture

The project uses **ResNet50** as the pretrained convolutional backbone.

ResNet50 is initialized with weights pretrained on ImageNet and is loaded without its original classification head. This allows the convolutional portion of the network to act as a general-purpose visual feature extractor.

A new task-specific classification head is added on top of the pretrained backbone.

The resulting architecture is:

```text
Input Image
     |
     v
ResNet50 Pretrained Backbone
     |
     v
Global Average Pooling 2D
     |
     v
Dense Layer
128 neurons
ReLU activation
     |
     v
Dropout
0.3
     |
     v
Dense Output Layer
1 neuron
Sigmoid activation
     |
     v
Cat / Dog Prediction
