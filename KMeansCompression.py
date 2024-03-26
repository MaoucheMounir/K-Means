from KMeans import *

class KMeansCompression(KMeans):
    def __init__(self, K:int, data:np.ndarray):
        """
        Args:
            K (int): Nb Clusters
            data (np.ndarray): matrice des données de forme (nb_echantillons, nb_dims)
        """
        KMeans.__init__(self, K, data)

    
    