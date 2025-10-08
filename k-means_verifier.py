import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# --- Paths to your saved model files ---
MODEL_PATH = "models/kmeans_model.pkl"
VECTORIZER_PATH = "models/kmeans_vectorizer.pkl"

print("🔍 Loading model and vectorizer...")

# --- Load model and vectorizer ---
with open(MODEL_PATH, "rb") as f:
    kmeans = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model and vectorizer loaded successfully.\n")

# --- Verify model structure ---
print("🧩 MODEL INFORMATION")
print(f"Type: {type(kmeans)}")
print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Iterations until convergence: {kmeans.n_iter_}")
print(f"Cluster center shape: {kmeans.cluster_centers_.shape}")

# --- Cluster label distribution ---
labels, counts = np.unique(kmeans.labels_, return_counts=True)
distribution = dict(zip(labels, counts))
print(f"\nCluster distribution: {distribution}")

# --- Check for degenerate clusters (e.g., all data fell into one) ---
if len(distribution) == 1:
    print("⚠️ Warning: The model collapsed into a single cluster.")
else:
    print("✅ Multiple clusters detected — good sign.")

# --- Verify vectorizer ---
print("\n🧠 VECTORIZER INFORMATION")
print(f"Type: {type(vectorizer)}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"n-gram range: {vectorizer.ngram_range}")
print(f"Stop words: {vectorizer.stop_words}")
print(f"Max features: {vectorizer.max_features}")

# --- Show top words per cluster ---
print("\n🔥 TOP TERMS PER CLUSTER:")
try:
    # Invert the vocabulary to get words from indices
    terms = np.array(sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get))
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    for i in range(kmeans.n_clusters):
        top_terms = terms[order_centroids[i, :10]]
        print(f"Cluster {i}: {', '.join(top_terms)}")

except Exception as e:
    print("⚠️ Could not extract top terms:", e)

# --- Final report ---
print("\n✅ Model verification complete.")
