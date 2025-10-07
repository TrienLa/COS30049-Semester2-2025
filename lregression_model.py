import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_utils import load_data, data_clean_up, data_preprocessing
from data_visualization import conf_matrix
import pickle
import sys, os

# Folder to save results
os.makedirs("processed/models", exist_ok=True)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline with TfidfVectorizer and LogisticRegression
    LRegression_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Model train
    LRegression_pipeline.fit(X_train, y_train)

    # Save model to a pickle file so we can use it later
    with open('processed/models/lr_model.pkl', 'wb') as picklefile:
        pickle.dump(LRegression_pipeline, picklefile)

    # Gerate a Confusion Matrix from available data
    conf_matrix('lr_model', X_test, y_test)

if __name__ == "__main__":
    # Load the email data
    email_dfs = pd.read_csv(sys.path[0] + '/dataset/emails.csv')
    data_clean_up(email_dfs)
    data_preprocessing(email_dfs)
    generate_model(data_fix(email_dfs))
