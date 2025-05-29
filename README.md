# Movie Review Sentiment Analysis

Movie Review Sentiment Analysis is a deep learning project built using Python and Keras. It is designed to classify the sentiment (positive or negative) of movie reviews using an LSTM-based neural network. The project includes complete workflows for text preprocessing, model training, evaluation, and result visualization.

---

## Overview

This project processes raw movie review text data to prepare it for training a sentiment classifier. It leverages NLP techniques such as tokenization, lemmatization, and stopword removal. The cleaned data is used to train an LSTM-based model that learns to detect sentiment polarity. The model’s performance is evaluated using accuracy, loss graphs, and classification metrics.

---

## Key Features

### Text Preprocessing
- Cleans review text using regular expressions, stopword removal, and lemmatization.
- Converts raw data into a format suitable for neural network input using tokenization and padding.

### Model Architecture
- A Sequential model built with embedding layers and a bidirectional LSTM layer.
- Includes dropout and L2 regularization to prevent overfitting.
- Final layer uses sigmoid activation for binary sentiment classification.

### Model Training
- Trains the model using training and validation data.
- Uses binary crossentropy loss and accuracy metrics.
- Includes adjustable hyperparameters like learning rate and dropout rate.

### Evaluation and Visualization
- Provides classification report and confusion matrix for test data.
- Includes training history plots for loss and accuracy across epochs.

### Reusable Code Modules
- `model.py`, `preprocess.py`, and `utils.py` provide clean, modular code structure.
- Notebooks (`.ipynb`) guide the end-to-end pipeline from data to results.

---

## Technology Stack

This project is built using the following technologies:

- **Python**: Core programming language.
- **Keras (TensorFlow backend)**: For building and training the neural network.
- **NLTK**: For natural language preprocessing.
- **Scikit-learn**: For evaluation metrics.
- **Matplotlib**: For plotting training history and confusion matrices.
- **Pandas**: For data manipulation and processing.

