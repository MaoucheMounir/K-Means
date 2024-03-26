import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from KMeans import *

class KMeans2d(KMeans):
    def __init__(self, K:int, data:np.ndarray):
        """
        Args:
            K (int): Nb Clusters
            data (np.ndarray): matrice des données de forme (nb_echantillons, nb_dims)
        """
        KMeans.__init__(self, K, data)

    def display_clusters(self, iteration=None) -> None:
        couleurs = list(mcolors.TABLEAU_COLORS.keys())
        for i, cluster in enumerate(self.clusters):
            points = cluster.data.tolist()
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            
            plt.scatter(x, y, color=couleurs[i], label=cluster.id_cluster)
        plt.scatter(self.centroids[:,0], self.centroids[:,1], color='yellow', label='centroids')
        plt.legend()
        if iter is not None:
            plt.title(f'{iteration=}')
        plt.show()
    
    def fit(self, nb_iter=10):
        self.display_clusters(iteration='init')
        for i in range(nb_iter):
            self.step()
            self.display_clusters(iteration=i+1)
            if self.verify_convergence():
                break
    
 

            
