import numpy as np
import pandas as pd

class Cluster():
    
    def __init__(self, data, couleur):
        self.data = data
        self.x, self.y = self.set_data(data)
        self.centroid = self.calculer_barycentre()
        self.couleur = couleur
        
    def set_data(self, data):
        x = data.iloc[:,0]
        y = data.iloc[:, 1]
        return x, y

    def calculer_barycentre(self):
        return np.array(np.mean(self.data, axis=0))
    