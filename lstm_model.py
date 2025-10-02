import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization
from tensorflow.python.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_processing import load_data, data_clean_up, data_preprocessing
import sys

# Load data and processing
email_dfs = load_data(sys.path[0] + "/dataset/emails.csv")
data_clean_up(email_dfs)
data_preprocessing(email_dfs)

# Tokenizer and pad sequences for model
vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=100,      # Ensure all sequences have the same length
    standardize='lower_and_strip_punctuation'   # Lowercase and split by whitespace
)
vectorizer.adapt(email_dfs['text'])
vectorized = vectorizer(email_dfs['text'])
X = vectorized.numpy()
y = to_categorical(email_dfs['spam'])

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model buidling and compiling
lstm = Sequential().add(Embedding(
                        input_dim=100,
                        output_dim=128)).add(LSTM(128)).add(Dense(2, activation='sigmoid'))

lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
lstm.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Model evaluation
loss, accuracy = lstm.evaluate(X_test, y_test)
print('LSTM Loss:', loss)
print('LSTM Accuracy:',accuracy)