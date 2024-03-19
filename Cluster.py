import numpy as np
import pandas as pd

class Cluster():
    
    def __init__(self, data:pd.Series, couleur:pd.Series):
        self.data = data
        
        
        #self.x, self.y = data.iloc[:,0], data.iloc[:,1]
        self.centroid = self.calculer_barycentre()
        self.couleur = couleur

    def calculer_barycentre(self):
        return np.array(np.mean(self.data, axis=0))
    