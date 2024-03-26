from sklearn.datasets import make_blobs
from KMeans2d import KMeans2d as KMeans


K = 3
nb_echantillons = 100
X, Y = make_blobs(n_samples=nb_echantillons, centers=K, n_features=2, cluster_std=0.5 , random_state=0)

km = KMeans(K,X)
km.fit()
print(km.inertie_normalisee())

