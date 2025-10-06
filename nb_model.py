import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_processing import load_data, data_clean_up, data_preprocessing
from data_visualization import conf_matrix
import pickle
import sys, os

# Folder to save results
os.makedirs("models", exist_ok=True)

def data_fix(email_df):
    """
    Args:
        email_df (pandas DataFrame): DataFrame containing information from CSV file.
    Returns:
        pandas. DataFrame: A DataFrame containing emails data without NaN values.
    """
    email_df_fix = email_df.dropna(subset=['text', 'spam']).copy()
    email_df_fix['spam'] = email_df_fix['spam'].astype(int)
    return email_df_fix

def generate_model(email_df):
    """
    Args:
        email_df (pandas DataFrame): DataFrame containing information from CSV file.
    """
    # Set the X, y for model training 
    X = email_df["text"]
    y = email_df["spam"]

    # Split dataset to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create pipeline with CountVectorizer and MultinomialNB
    NaiveBayes_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', GaussianNB())
    ])

    # Model train
    NaiveBayes_pipeline.fit(X_train, y_train)

    # Save model to a pickle file so we can use it later
    with open('processed/models/nb_classifier.pkl', 'wb') as picklefile:
        pickle.dump(NaiveBayes_pipeline, picklefile)

    # Gerate a Confusion Matrix from available data
    conf_matrix("nb_classifier", X_test, y_test)

if __name__ == "__main__":
    # Load the email data
    email_dfs = pd.read_csv(sys.path[0] + 'dataset/emails.csv')
    data_clean_up(email_dfs)
    data_preprocessing(email_dfs)
    generate_model(data_fix(email_dfs))

    # Test prediction
    with open('processed/models/nb_classifier.pkl', 'rb') as tm:
        new_pipe = pickle.load(tm)
        print(new_pipe.predict(['Subject hi im a fake email and this is a scam']))