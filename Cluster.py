import numpy as np
import pandas as pd

class Cluster():
    
    def __init__(self, data:pd.Series, id_cluster:pd.Series):
        self.data = data
        
        #self.x, self.y = data.iloc[:,0], data.iloc[:,1]
        self.centroid:np.ndarray = self.calculer_barycentre()
        self.couleur = id_cluster

    def calculer_barycentre(self) -> np.ndarray:
        return np.array(np.mean(self.data, axis=0))
    