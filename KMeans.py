from matplotlib import pyplot as plt
import random as rd
import numpy as np
import pandas as pd

from Cluster import Cluster

class KMeans():
    def __init__(self, K:int, data:np.ndarray):
        """
        Args:
            K (int): Nb Clusters
            data (np.ndarray): matrice des données de forme (nb_echantillons, nb_dims)
        """
        self.data = data
        #self.x, self.y = data[:,0], data[:,1]
        self.nb_echantillons = len(data)
        
        self.centroids:np.ndarray = self.init_centroids(K, data)
        self.indices_clusters = np.arange(K)
        self.clusters:list[Cluster] = None
        self.previous_centroids = None

    def init_centroids(self, K:int, data:np.ndarray) -> np.ndarray:
        centroids_indices = [rd.randint(0, self.nb_echantillons) for i in range(K)]
        centroids = np.array([data[indice] for indice in centroids_indices])
        
        return centroids

    def get_nearest_centroid(self, point:np.ndarray) -> int:
        """calcule la distance euclidienne entre un point et tous les centroids et retourne le centroid argmin

        Args:
            point (np.ndarray): un point

        Returns:
            _type_: l'indice du centroid
        """
        distances = []

        for c in self.centroids:
            distances.append(np.sqrt(np.sum((point-c)**2)))

        return distances.index(np.min(distances))

    def assign_nearest_centroid(self) -> list[tuple]:
        # Assign each data point to its nearest centroid
        points_centroids = []
        
        for point in self.data:
            
            points_centroids.append(self.get_nearest_centroid(point))
        
        #z = list(zip(self.x, self.y, points_centroids))
        z = list(zip(self.data, points_centroids))
        return z
    
    def create_clusters(self, z) -> None:
        df = pd.DataFrame(z, columns=['point', 'cluster'])
        groupes = df.groupby('cluster')

        self.clusters = [Cluster(df_groupe.iloc[:,0], indice_cluster) for indice_cluster, df_groupe in groupes]
    
    def update_centroids(self,) -> None: #Le calcul du barycentre se fait automatiquement lors de la creation des clusters
        self.centroids = np.array([cluster.centroid for cluster in self.clusters])

    def step(self,) -> None:
        z = self.assign_nearest_centroid()
        
        self.create_clusters(z)    
        self.previous_centroids = self.centroids.copy()
        self.update_centroids()
    
    def verify_convergence(self) -> bool:
        return (np.array(self.centroids) - np.array(self.previous_centroids)).all() < 1e-5
    
    def fit(self, nb_iter=10) -> None:
        for i in range(nb_iter):
            self.step()
            #self.display_clusters()
            if self.verify_convergence():
                break
    
    def predict(self, point) -> int:
        return self.get_nearest_centroid(point)
    
    def transform_pixel(self, point:np.ndarray) -> np.ndarray:
        """transforme un point (qui doit exister dans l'ensemble de train)
            vers sa couleur compressée
        Args:
            point (np.ndarray): les coordonnées RGB du point de départ
        Returns:
            np.ndarray: les nouvelles coordonnées RGB du point
        """
        for c in self.clusters:
            for p in c.data: 
                if np.array_equal(p, point):
                    return c.centroid 