import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import transformation

def kmeans(df, n_clusters = 0, normalize = True):
    """
    Make unsupervised KMeans clustering (sklearn.clusert.KMeans).

    df --- pandas.DataFrame
    n_clusters --- number of clusters in model (default = 0)
    normalize --- should all values be normalized (defauld = True)

    return --- sklearn.clusert.KMeans
    """

    if (normalize):
        df = transformation.StandardScale(df)

    model = KMeans(n_clusters)
    model.fit(df)
    model.predict(df)

    return model