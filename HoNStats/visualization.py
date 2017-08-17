import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

from sklearn import preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates, andrews_curves

def scaleFeaturesDF(df):
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)

    return scaled
def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax

def pca_visualize_2d(df, scaleFeatures = False, normalize = True):
    if scaleFeatures: df = scaleFeaturesDF(df)

    if normalize: 
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(np_scaled)
        print(df)

    pca = PCA(2)
    pca.fit(df)

    transformed = pca.transform(df)
    transformed  = pd.DataFrame(transformed)

    transformed.columns = ['component1', 'component2']

    ax, fig = plt.subplots()

    if normalize: fig.set_title("Normalized 2D Scatter")
    else: fig.set_title("Non-normalized 2D Scatter")

    plt.scatter(transformed['component1'], transformed['component2'], marker='o', alpha=0.6)

    plt.draw()
def pca_visualize_3d(df, scaleFeatures = False, normalize = True):
    if scaleFeatures: df = scaleFeaturesDF(df)

    if normalize: 
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(np_scaled)
        print(df)

    pca = PCA(3)
    pca.fit(df)

    transformed = pca.transform(df)


    fig = plt.figure()
    ax = Axes3D(fig)
    
    if normalize: ax.set_title("Normalized 3D Scatter")
    else: ax.set_title("Non-normalized 3D Scatter")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2] , c='blue', marker='.')

    plt.draw()

def parallel_coords(df, class_column, normalize = False):
    plt.style.use("ggplot")
    title = "Non-normalized Parallel Coordinates"

    if normalize: 
        titles = list(df)
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(np_scaled, columns = titles)
        title = "Normalized Parallel Coordinates"

    plt.figure()
    plt.title(title)
    parallel_coordinates(df, class_column = class_column, cols = list(df), alpha = 0.4)
    plt.draw()

def andrews_curves(df, class_column, normalize = False):
    plt.style.use("ggplot")
    title = "Non-normalized Andrew's Curves"

    if normalize: 
        titles = list(df)
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(np_scaled, columns = titles)
        title = "Normalized Andrew's Curves"
    
    plt.figure()
    plt.title(title)
    pd.plotting.andrews_curves(df, class_column)
    plt.draw()

def draw_all():
    plt.show()