import pandas as pd
import numpy as np
from sklearn import preprocessing, manifold

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

def isomap(df, n_components, n_neighbours, normalize = True):
    if normalize:
        df = StandardScale(df)

    iso = manifold.Isomap(n_neighbours, n_components)
    iso.fit(df)

    return iso.transform(df)