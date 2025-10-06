import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from data_processing import load_data, data_clean_up, data_preprocessing
from data_visualization import conf_matrix
import os
import pickle

# Folder to save results
os.makedirs("processed/models", exist_ok=True)

def extract_features(email_df):
    """
    Converts email text data into numerical TF-IDF feature vectors.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(email_df['text'])
    return X, vectorizer

def generate_kmeans_model(email_df, n_clusters=2):
    """
    Creates and evaluates a K-Means clustering model from email data.
    """
    # Extract Features
    print("Extracting features...")
    X, vectorizer = extract_features(email_df)
    print("Training K-Means model...")

    # Create and train K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # Attach results to DataFrame
    email_df['cluster'] = kmeans.labels_

    # Evaluate if labels exist
    if 'spam' in email_df.columns:
        ari = adjusted_rand_score(email_df['spam'], kmeans.labels_)
        silhouette = silhouette_score(X, kmeans.labels_)
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")

    # Save model and vectorizer
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "processed", "models", "kmeans_model.pkl")
    vectorizer_path = os.path.join(base_dir, "processed", "models", "kmeans_vectorizer.pkl")

    with open(model_path, "wb") as model_file:
        pickle.dump(kmeans, model_file)
    with open(vectorizer_path, "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print(f"K-Means model saved successfully to: {model_path}")

    # Gerate a Confusion Matrix from available data

    return kmeans


if __name__ == "__main__":
    # Step 1: Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
    dataset_path = os.path.join(base_dir, "dataset", "emails.csv")

    # sanity check
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Step 2: Clean and preprocess
    email_df = load_data(dataset_path)
    data_clean_up(email_df)
    data_preprocessing(email_df)

    # Step 3: Drop missing text
    email_df = email_df.dropna(subset=['text'])

    # Step 4: Generate and save K-Means model
    kmeans = generate_kmeans_model(email_df)
