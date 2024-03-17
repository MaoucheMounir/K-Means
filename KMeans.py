from matplotlib import pyplot as plt
import random as rd
import numpy as np
import pandas as pd

from Cluster import Cluster

class KMeans():
    def __init__(self, K, data):
        self.data = data
        self.x, self.y = self.set_data(data)
        self.nb_echantillons = len(data)
        self.centroids = self.init_centroids(K, data)
        self.colors = ['blue', 'green', 'red', 'black', 'brown']
        self.clusters = None
        self.previous_centroids = None

    def init_centroids(self, K, X):
        #Initialize centroids
        np.random.seed(0)
        centroids_indices = [rd.randint(0, self.nb_echantillons) for i in range(K)]
        centroids = pd.DataFrame([(X[indice, 0], X[indice, 1]) for indice in centroids_indices], columns=['x', 'y'])
        
        return centroids

    def set_data(self, data):
        x = data[:,0]
        y = data[:, 1]
        
        return x, y

    def get_nearest_centroid(self, point):
        distances = []

        for c in np.array(self.centroids):
            distances.append(np.sum((point-c)**2))

        return distances.index(np.min(distances))

    def assign_nearest_centroid(self,):
        # Assign each data point to its nearest centroid
        point_colors = []
            
        for point in self.data:
            point_colors.append(self.colors[self.get_nearest_centroid(point)])

            
        z = list(zip(self.x, self.y, point_colors))
        return z
    
    def create_clusters(self, z):
        df = pd.DataFrame(z, columns=['x', 'y', 'couleur'])
        groupes = df.groupby('couleur')

        # Création de DataFrame pour chaque groupe
        # df_rouge = groupes.get_group('blue').iloc[:,:2]
        # df_bleu = groupes.get_group('green').iloc[:,:2]
        # df_vert = groupes.get_group('red').iloc[:,:2]
        #Cluster(df_groupe, couleur)
        self.clusters = [Cluster(df_groupe.iloc[:,:2], couleur) for couleur, df_groupe in groupes]
        
    def update_centroids(self,):
        self.centroids = np.array([cluster.centroid for cluster in self.clusters])

    def step(self,):
        z = self.assign_nearest_centroid()
        self.create_clusters(z)    
        self.previous_centroids = self.centroids.copy()
        self.update_centroids()
        
    def display_clusters(self):
        for cluster in self.clusters:
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
            self.display_clusters()
            if self.verify_convergence():
                break
    
    def predict(self, point):
        return self.get_nearest_centroid(point)
    