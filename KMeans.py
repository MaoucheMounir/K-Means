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

    def init_centroids(self, K:int, data:np.ndarray):
        #Initialize centroids
        np.random.seed(0)
        centroids_indices = [rd.randint(0, self.nb_echantillons) for i in range(K)]
        
        #centroids = np.array([(X[indice, 0], X[indice, 1]) for indice in centroids_indices])
        centroids = np.array([data[indice] for indice in centroids_indices])
        
        return centroids

    def get_nearest_centroid(self, point:np.ndarray):
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

    def assign_nearest_centroid(self):
        # Assign each data point to its nearest centroid
        points_centroids = []
        
        for point in self.data:
            
            points_centroids.append(self.get_nearest_centroid(point))
        
        #z = list(zip(self.x, self.y, points_centroids))
        z = list(zip(self.data, points_centroids))
        return z
    
    """# def assign_nearest_centroid(self):
    #     # Assign each data point to its nearest centroid
    #     point_colors = []
            
    #     for point in self.data:
    #         point_colors.append(self.colors[self.get_nearest_centroid(point)])
            
    #     z = list(zip(self.x, self.y, point_colors))
    #     return z"""
    
    def create_clusters(self, z):
        df = pd.DataFrame(z, columns=['point', 'cluster'])
        #print(df)
        groupes = df.groupby('cluster')

        self.clusters = [Cluster(df_groupe.iloc[:,0], indice_cluster) for indice_cluster, df_groupe in groupes]
        
    def update_centroids(self,): #Le calcul du barycentre se fait automatiquement lors de la creation des clusters
        self.centroids = np.array([cluster.centroid for cluster in self.clusters])

    def step(self,):
        z = self.assign_nearest_centroid()
        #print(z)
        self.create_clusters(z)    
        self.previous_centroids = self.centroids.copy()
        self.update_centroids()
        
    def display_clusters(self):
        for cluster in self.clusters:
            print(cluster.x)
            plt.scatter(cluster.x, cluster.y, color=cluster.couleur, label=cluster.couleur)
        plt.scatter(self.centroids[:,0], self.centroids[:,1], color='yellow', label='yellow')

        plt.show()
    
    def verify_convergence(self):
        if (np.array(self.centroids) - np.array(self.previous_centroids)).all() < 1e-5:
            return True
        else:
            return False
    
    def fit(self, nb_iter=10):
        
        for i in range(nb_iter):
            self.step()
            #self.display_clusters()
            if self.verify_convergence():
                break
    
    def predict(self, point):
        return self.get_nearest_centroid(point)
    
    def get_nouvelle_couleur_du_point(self, point):
        for c in self.clusters:
            for p in c.data: #.tolist():
                if np.array_equal(p, point):
                    return c.centroid 