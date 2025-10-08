import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix, classification_report

from data_utils import load_data, data_clean_up, data_preprocessing
from data_visualization import conf_matrix
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import os
import pickle
import sys

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

        # Confusion Matrix
        y_true = email_df['spam'].to_numpy()
        y_pred = map_clusters_to_labels(kmeans.labels_, y_true)

        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))

        # Optional: Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Ham", "Spam"],
                    yticklabels=["Ham", "Spam"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (K-Means Clustering)")
        os.makedirs("processed/confusion_matrix", exist_ok=True)
        plt.savefig("processed/confusion_matrix/kmeans_confusion_matrix.png", dpi=300)
        plt.show()

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

def map_clusters_to_labels(cluster_labels, true_labels):
    """
    Map each KMeans cluster to the most frequent true label within it.
    """
    mapped = np.zeros_like(cluster_labels)
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        mapped[mask] = mode(true_labels[mask])[0]
    return mapped

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
    email_df = pd.read_csv(sys.path[0] + '/dataset/combined_dataset.csv')
    email_df = data_clean_up(email_df)
    email_df = data_preprocessing(email_df)

    # Step 4: Generate and save K-Means model
    kmeans = generate_kmeans_model(email_df)


