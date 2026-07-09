# SMS Spam Detection using LSTM

This project implements an SMS spam detection system using a Long Short-Term Memory (LSTM) neural network. The goal is to classify SMS messages as either 'ham' (legitimate) or 'spam' based on their textual content.

1. Project Overview

The notebook demonstrates the complete workflow for building a text classification model:

Data Loading and Initial Exploration: Loading the dataset and examining its structure.
Data Preprocessing: Cleaning text data, handling stopwords, stemming, and converting text into numerical representations suitable for a neural network.
Model Building: Constructing an LSTM-based deep learning model using TensorFlow/Keras.
Training and Evaluation: Training the model on preprocessed data and evaluating its performance using standard metrics.
2. Dataset

The dataset used is spam.csv, which contains SMS messages labeled as 'ham' or 'spam'.

Source: /content/spam.csv
Columns: Category (label) and Message (SMS text).
Shape: (5572, 2)
3. Dependencies

Key libraries used in this project include:

pandas for data manipulation.
nltk for natural language processing tasks (stopwords, stemming).
tensorflow.keras for building and training the LSTM model.
sklearn for data splitting and model evaluation metrics.
4. Data Preprocessing Steps

Load Data: The spam.csv file is loaded into a pandas DataFrame.
Target Encoding: The Category column is converted from categorical strings ('ham', 'spam') to numerical labels (0 for 'ham', 1 for 'spam').
Text Cleaning (corpus creation):
Messages are converted to lowercase.
Non-alphabetic characters are removed.
Messages are tokenized into words.
Stop words (common words like 'the', 'is', 'a') are removed using nltk.corpus.stopwords.
Words are stemmed using PorterStemmer to reduce them to their root form (e.g., 'running' -> 'run').
One-Hot Encoding: Each word in the cleaned corpus is converted into a numerical representation using one_hot encoding, based on a vocabulary size (voc_size = 5000).
Padding Sequences: The one-hot encoded sequences are padded to a fixed maximum length (max_length = 30) to ensure uniform input size for the neural network. Padding is applied post-sequence.
5. Model Architecture (LSTM)

The model is a sequential Keras model with the following layers:

Embedding Layer: Maps integer-encoded words to dense vectors of fixed size (embedding_vector_features = 40).
model.add(Embedding(voc_size, embedding_vector_features, input_length=max_length))
Dropout Layer (0.3): Regularization layer to prevent overfitting.
LSTM Layer: A Long Short-Term Memory layer with 150 units, capable of learning long-term dependencies in sequences.
model.add(LSTM(150))
Dropout Layer (0.3): Another regularization layer.
Dense Output Layer: A single neuron with a sigmoid activation function for binary classification.
model.add(Dense(1, activation='sigmoid'))
Compilation: The model is compiled with:

Loss Function: binary_crossentropy (suitable for binary classification).
Optimizer: adam.
Metrics: accuracy.
6. Training

The dataset is split into training and testing sets (80% training, 20% testing) using train_test_split with random_state=45.

The model is trained for a specified number of epochs (initially 10, then 2 in a refined model) with a batch_size of 64.

7. Evaluation

After training, the model's performance is evaluated on the test set:

Predictions: y_pred = model.predict(X_test) generates raw probability scores.
Thresholding: Predictions are converted to binary classes (0 or 1) using a threshold of 0.6 (np.where(y_pred > 0.6, 1, 0)).
Accuracy Score: accuracy_score(y_pred, y_test) calculates the overall accuracy.
Classification Report: classification_report(y_pred, y_test) provides detailed metrics like precision, recall, and f1-score for each class.
8. Results

The model achieved an accuracy of approximately 98.03% on the test set. The classification report indicates strong performance, especially for the 'ham' class (0). The 'spam' class (1) also shows good recall, although precision is slightly lower, which is common in imbalanced datasets like spam detection where false positives are more tolerated than false negatives for spam.

Classification Report:

              precision    recall  f1-score   support

           0       1.00      0.98      0.99       984
           1       0.87      0.98      0.92       131

    accuracy                           0.98      1115
   macro avg       0.93      0.98      0.96      1115
weighted avg       0.98      0.98      0.98      1115
9. Usage

To run this project, simply execute the cells in the notebook sequentially. Ensure all necessary libraries are installed (pandas, nltk, tensorflow, sklearn). The nltk stopwords will be downloaded automatically if not already present.
