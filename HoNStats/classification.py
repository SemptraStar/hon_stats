import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import transformation

def kneighbors(n_samples, n_outputs, n_neighbors=3):
    """Peform K-Neighbors classification.

    n_samples - data samples
    n_outputs - classes for data samples
    n_neighbors - number of neighbors for each sample.
    Keep it odd when doing binary classification, particularly when you use uniform weighting.

    """

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(n_samples, n_outputs)
    return model

