import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_utils import load_data, data_clean_up, data_preprocessing
from data_visualization import conf_matrix
import pickle
import sys, os

# Folder to save results
os.makedirs("processed/models", exist_ok=True)

def generate_model(email_df):
    """
    Args:
        email_df (pandas DataFrame): DataFrame containing information from CSV file.
    """
    # Set the X, y for model training 
    X = email_df["text"]
    y = email_df["spam"]

    # Split dataset to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline with TfidfVectorizer and MultinomialNB
    NaiveBayes_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('classifier', MultinomialNB())
    ])

    # Model train
    NaiveBayes_pipeline.fit(X_train, y_train)

    # Save model to a pickle file so we can use it later
    with open('processed/models/nb_model.pkl', 'wb') as picklefile:
        pickle.dump(NaiveBayes_pipeline, picklefile)

    # Gerate a Confusion Matrix from available data
    conf_matrix('nb_model', X_test, y_test)

if __name__ == "__main__":
    # Load the email data
    email_dfs = pd.read_csv(sys.path[0] + '/dataset/combined_dataset.csv')
    data_clean_up(email_dfs)
    data_preprocessing(email_dfs)
    generate_model(email_dfs)
