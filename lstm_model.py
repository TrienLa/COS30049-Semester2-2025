import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.python import keras
import keras.models as kmodels
from keras.models import Sequential
from keras.layers import TextVectorization
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_processing import load_data, data_clean_up, data_preprocessing
import sys, os

# Folder to save results
os.makedirs("models", exist_ok=True)

def generate_model(email_df):
    """
    Args:
        email_df (pandas DataFrame): DataFrame containing information from CSV file.
    """
    # Tokenizer and pad sequences for model
    vectorizer = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=100,      # Ensure all sequences have the same length
        standardize='lower_and_strip_punctuation'   # Lowercase and split by whitespace
    )
    vectorizer.adapt(email_df['text'])
    vectorized = vectorizer(email_df['text'])
    X = vectorized.numpy()
    y = to_categorical(email_df['spam'])

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

    # Save model to a keras (tensorflow keras model extension) file so we can use it later
    lstm.save('models/lstm.keras')

if __name__ == "__main__":
    # Load data and processing
    email_dfs = load_data(sys.path[0] + "/dataset/emails.csv")
    data_clean_up(email_dfs)
    data_preprocessing(email_dfs)
    generate_model(email_dfs)

    # Load model
    model = kmodels.load_model(sys.path[0] +'/models/lstm_model.keras')
