from sklearn.datasets import make_blobs
from KMeans import KMeans
import numpy as np

K = 5
nb_echantillons = 100
X, Y = make_blobs(n_samples=nb_echantillons, centers=K, n_features=2, cluster_std=0.5 , random_state=0)

km = KMeans(K,X)
km.fit()

