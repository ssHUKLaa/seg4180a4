from datasets import load_dataset
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
dataset = load_dataset("sh0416/ag_news", split="train")

df = pd.DataFrame(dataset)

df = df.sample(n=5000, random_state=42).reset_index(drop=True)

texts = (df["title"] + " " + df["description"]).values
#get features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(texts)
inertia = []
K = range(2, 11)
#det optimal k
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()
#seems like elbow at k=7
k = 7
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df["cluster"] = clusters

pca = PCA(n_components=2, random_state=42)
X_dense = X.toarray()
X_scaled = StandardScaler(with_mean=False).fit_transform(X_dense)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=clusters,
    palette="tab10",
    s=40
)
plt.title("K-Means Clusters (PCA Projection)")
plt.show()

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(k):
    print(f"\nCluster {i}:")
    for j in order_centroids[i, :10]:
        print(terms[j], end=", ")

X_train, X_test, y_train, y_test = train_test_split(
    X, clusters, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))