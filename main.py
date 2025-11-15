# TfidfVectorizer converts text documents into a numerical vector (TF-IDF matrix).
# KMeans is the clustering algorithm.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. Sample Corpus
# You created a corpus of 7 documents.
# First 4 are about AI, ML, Python (technology).
# Last 3 are about sports.
# The goal is to automatically group these into clusters
documents = [
    "Machine learning is fascinating. It allows computers to learn from data.",
    "Deep learning is a subset of machine learning that uses neural networks.",
    "Python is a great programming language for data science.",
    "Artificial intelligence is the future of technology.",
    "Basketball is a popular sport played worldwide.",
    "Football and cricket are famous outdoor sports.",
    "Sports like tennis require great stamina and skill."
]

# 2. Preprocessing + TF-IDF
# Converts text → numerical matrix.
# stop_words='english' removes common words like is, the, a, from.
# X becomes a TF-IDF matrix where:
# Rows = documents
# Columns = unique words
# Values = importance of words
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

print("TF-IDF Matrix Shape:", X.shape)


# 3. K-Means
# KMeans tries to divide data into 2 clusters (because k = 2).
# It looks at word patterns to decide groups.
k = 2
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

# Print labels for each document
labels = model.labels_
print("\nCluster labels:")
for i, label in enumerate(labels):
    print(f"Document {i}: Cluster {label}")

# 4. Top Terms
# terms → list of all unique words in vocab
# model.cluster_centers_ → each cluster has a vector (centroid)
# argsort() sorts each cluster's word weights
# [:, ::-1] reverses order (largest → smallest)
terms = vectorizer.get_feature_names_out()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

# Print top 10 words for each cluster
print("\nTop terms per cluster:")
for cluster in range(k):
    print(f"\nCluster {cluster}:")
    for ind in order_centroids[cluster, :10]:
        print(terms[ind])
