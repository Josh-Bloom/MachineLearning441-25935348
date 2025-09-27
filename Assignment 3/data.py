from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import numpy as np

def fetch_data(id):
    ids = {
        "glass": 42,
        "sil": 149,
        "car": 19
    }
    # fetch dataset 
    glass_identification = fetch_ucirepo(id=ids[id]) 
  
    # data (as pandas dataframes) 
    X = glass_identification.data.features 
    y = glass_identification.data.targets
    # detect if the dataset is balanced (and make imbalanced if so)
    X = StandardScaler().fit_transform(X)
    y = np.array(y)
    return X, y