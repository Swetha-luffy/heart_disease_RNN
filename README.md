# Academics
# Parametrically Optimized Automated Diagnostic System for Heart Disease Prediction using Deep Neural Networks

This project focuses on building deep learning models to predict heart disease using patient data. 
Two models, a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN), are implemented to classify the presence of heart disease based on medical features such as age, sex, chest pain type, resting blood pressure, and others.

# Project Overview

The goal of this project is to compare the performance of CNN and RNN models in accurately predicting heart disease using a binary classification approach. Both models were evaluated using validation data, with performance metrics including accuracy, precision, recall, and F1-score.

# Key Features:

CNN Model: Utilizes 1D convolutional layers to extract features from structured data.

RNN Model: Leverages sequential processing power of RNNs to capture patterns in the dataset.

Data Preprocessing: Includes reshaping and encoding target variables for compatibility with neural network architectures.

Evaluation Metrics: Accuracy, precision, recall, F1-score, and confusion matrices to assess model performance.

Model Comparison: Visual comparisons of CNN and RNN model performance are presented.

# Dataset
The dataset used contains various health-related parameters such as:

Age

Gender

Chest pain type (cp)

Resting blood pressure (trestbps)

Maximum heart rate achieved (thalach)

Exercise induced angina (exang)

...and more.

Data is split into training and testing sets to evaluate the models.

# Models

 Convolutional Neural Network (CNN)

2.convolutional layers with max pooling and dropout for regularization.

3.Activation: ReLU in hidden layers and Sigmoid in the output layer.

4.Validation Accuracy: ~93.5%

Recurrent Neural Network (RNN)

1 SimpleRNN layers with dropout for regularization.

2 Activation: ReLU in hidden layers and Sigmoid in the output layer.

3 Validation Accuracy: ~98.5%


# Model Training

Batch Size: 16

Epochs: 500

Optimizer: Adam

Loss Function: Binary Crossentropy

# Visualization
Both accuracy and loss during the training process were plotted to analyze model convergence. Confusion matrices for both CNN and RNN are plotted to display the performance on the test data.

# Results

CNN Model achieved a validation accuracy of 93.5%.

RNN Model achieved a validation accuracy of 98.5%.

RNN outperformed CNN in both precision and recall, making it the more effective model for heart disease prediction in this dataset.

# Requirements

Python 3.x
TensorFlow 2.x
Numpy
Pandas
Matplotlib
Seaborn
Usage
Clone the repository:


Run the Jupyter notebooks for preprocessing and model training:

1_Preprocessing_File.ipynb
2_ModelTrainingFile.ipynb
Evaluate the models using the provided test data.

License
This project is licensed under the MIT License - see the LICENSE file for details.
