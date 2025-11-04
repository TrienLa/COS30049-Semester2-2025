import keras.models as kmodels
import tensorflow as tf
from keras.models import Sequential
from keras.layers import TextVectorization, Normalization
from keras.layers import Embedding, LSTM, Dense
from data_utils import load_data, data_clean_up, data_preprocessing
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
        max_tokens=1000,
        output_mode='int',
        output_sequence_length=128,
        standardize='lower_and_strip_punctuation'
    )
    vectorizer.adapt(email_df['text'])
    
    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(email_df['text']),tf.constant(email_df['spam'])))
    ds = ds.shuffle(buffer_size=8).batch(2).prefetch(tf.data.AUTOTUNE)

    # Model buidling and compiling
    lstm = Sequential([
        vectorizer,
        Embedding(input_dim=1000, output_dim=128, mask_zero=True),
        LSTM(128),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
        ])

    lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Model training
    lstm.fit(ds, epochs=10)

    # Model evaluation
    loss, accuracy = lstm.evaluate(ds)
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
    model = kmodels.load_model(sys.path[0] +'/models/lstm.keras')

    # Test data
    test_mails = [
        'hi im a fake email and this is a scam',
        'naturally irresistible your corporate identity  lt is really hard to recollect a company : the  market is full of suqgestions and the information isoverwhelminq ; but a good  catchy logo , stylish statlonery and outstanding website  will make the task much easier .  we do not promise that havinq ordered a iogo your  company will automaticaily become a world ieader : it isguite ciear that  without good products , effective business organization and practicable aim it  will be hotat nowadays market',
    ]

    # Model Prediction
    predictions = model.predict(tf.constant(test_mails))

    print(predictions)
