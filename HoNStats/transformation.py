import pandas as pd
import numpy as np
from sklearn import preprocessing, manifold
from sklearn import decomposition

def StandardScale(df):
    """
    Normalize all values in DataFrame (set it between [0.0, 1.0])

    df --- pandas.DataFrame

    return --- normalized pandas.DataFrame
    """
    titles = list(df)
    scaler = preprocessing.StandardScaler()
    np_scaled = scaler.fit_transform(df)

    return pd.DataFrame(np_scaled, columns = titles)

def pca(df, n_components, normalize=True, labels=[]):
    """Perform PCA on given DataFrame"""

    if normalize:
        df = StandardScale(df)

    pca = decomposition.PCA(n_components)
    transformed = pd.DataFrame(pca.fit_transform(df))

    if labels:
        transformed.columns = labels
    else:
        transformed.columns = ["Component {0}".format(i) for i in range(transformed.shape[1])]

    return transformed
def isomap(df, n_components, n_neighbours, normalize=True, labels=[]):
    """Perform Isomap on given DataFrame"""

    if normalize:
        df = StandardScale(df)

    iso = manifold.Isomap(n_neighbours, n_components)
    transformed = pd.DataFrame(iso.fit_transform(df))

    if labels:
        transformed.columns = labels
    else:
        transformed.columns = ["Component {0}".format(i) for i in range(transformed.shape[1])]

    return transformed