import numpy as np
import pandas as pd

class Cluster():
    
    def __init__(self, data:np.ndarray, id_cluster:pd.Series):
        self.data = data
        
        #self.x, self.y = data.iloc[:,0], data.iloc[:,1]
        self.centroid:np.ndarray = self.calculer_barycentre()
        self.id_cluster:int = id_cluster

    def calculer_barycentre(self) -> np.ndarray:
        return np.mean(self.data, axis=0)
    