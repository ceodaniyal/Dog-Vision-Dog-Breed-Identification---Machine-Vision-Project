# Dog Vision: Dog Breed Identification - Machine Vision Project

## Overview

This project focuses on developing a machine vision system capable of identifying a dog's breed from a given image. Utilizing deep learning techniques, specifically Convolutional Neural Networks (CNNs), the model has been trained to classify images of dogs into multiple breeds. This multiple-image classifier leverages state-of-the-art neural networks to achieve high accuracy in breed identification.

## Dataset

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/zayl001/dog-breed-identification). It contains thousands of labeled images of dogs spanning over 120 different breeds. Each image is associated with a breed label, making it suitable for training and evaluating deep-learning models.

### Dataset Features
- **Number of Classes**: 120 dog breeds.
- **Image Size**: Varies, but typically resized to a consistent size during preprocessing.
- **Format**: JPEG images with corresponding breed labels.

## Model Architecture

The project leverages a pre-trained model from TensorFlow Hub, specifically the MobileNetV2 architecture. This model has been fine-tuned to cater to the specific task of dog breed identification, allowing it to generalize well across different breeds.

### Key Components
- **Input Shape**: The images are resized to a uniform shape of `[None, IMG_SIZE, IMG_SIZE, 3]` to match the input requirements of the MobileNetV2 model.
- **Output Layer**: The final layer of the model is adjusted to have 120 neurons, corresponding to the number of dog breeds.
- **Loss Function**: Categorical Cross entropy is used, which is standard for multi-class classification problems.
- **Optimizer**: Adam optimizer is utilized for its efficiency and adaptability.

## Training Process

The model is trained on the Kaggle dataset using a GPU for accelerated training. Various techniques such as data augmentation and learning rate scheduling are employed to enhance the model's performance and prevent overfitting.

### Training Highlights
- **Epochs**: The model is trained over multiple epochs, allowing it to learn the intricate features of different dog breeds.
- **Validation**: A portion of the dataset is set aside for validation to monitor the model's performance and adjust hyperparameters accordingly.

## Results

The final model achieves a high accuracy rate on the validation set, demonstrating its capability to accurately identify dog breeds from images. The trained model is saved and can be used to make predictions on new, unseen images.

## How to Use

1. **Clone the Repository**: Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/ceodaniyal/Dog-Vision-Dog-Breed-Identification---Machine-Vision-Project.git
2. **Dataset:** Download the dataset from [Kaggle](https://www.kaggle.com/datasets/zayl001/dog-breed-identification) and place it in the appropriate directory.
3. Training the Model: Run the training script to train the model on your machine or use the provided pre-trained model for inference.
4. Prediction: Use the inference script to predict the breed of a dog from a new image.
