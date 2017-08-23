import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
import transformation

from sklearn import preprocessing, manifold
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates, andrews_curves
from sklearn.cluster import KMeans


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

def StandardScale(df):
    titles = list(df)
    scaler = preprocessing.StandardScaler()
    np_scaled = scaler.fit_transform(df)
    return pd.DataFrame(np_scaled, columns = titles)

def scatter_2d(df, x, y, title = "Scatter plot 2D"):
    df.scatter(df.loc[:, x], df.loc[:, y], c = "b")
    plt.draw()

def cluster_scatter_2d(df, model, x, y, colors, xlabel = "X", ylabel = "Y", title = "Clastering Scatter plot 2D"):
    centers = model.cluster_centers_

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(len(centers)):
        ax.scatter(df[model.labels_ == i].iloc[:, x], df[model.labels_ == i].iloc[:, y], c=colors[i], marker = ".", alpha = "0.6")
        ax.scatter(centers[i, 0], centers[i, 1], s=169, c='#000000', marker='x', alpha=0.8, linewidths=2)
        ax.annotate("Center {0}".format(i), xy = (centers[i, 0], centers[i, 1]), size = 13)
    
    plt.draw()
def cluster_scatter_3d(df, model, colors, x = 0, y = 1, z = 2, xlabel = "X", ylabel = "Y", zlabel = "Z", title = "Clastering Scatter plot 3D", normalize = True):
    centers = model.cluster_centers_

    if normalize:
        df = transformation.StandardScale(df)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    for i in range(len(centers)):
        group = df[model.labels_ == i]
        ax.scatter(group.iloc[:, x], group.iloc[:, y], group.iloc[:, z], c=colors[i], marker = ".", alpha = 0.6)
        ax.scatter(centers[i, 0], centers[i, 1], centers[i, 2], s=169, c='#000000', marker='x', alpha=0.8, linewidths=2)        
    
    plt.draw()


def pca_2d(df, scaleFeatures = False, normalize = True):
    if scaleFeatures: df = scaleFeaturesDF(df)

    if normalize: 
        df = StandardScale(df)

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
def pca_3d(df, scaleFeatures = False, normalize = True):
    if scaleFeatures: df = scaleFeaturesDF(df)

    if normalize: 
        df = StandardScale(df)

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

def isomap_2d(df, n_neighbours):
    # Scale section
    df = StandardScale(df)

    # Isomap section
    iso = manifold.Isomap(n_neighbours, n_components = 2)
    iso.fit(df)

    transformed = iso.transform(df)

    # Visualization

    fig = plt.figure()
    plt.title("Isomap 2D")
    plt.scatter(transformed[:,0], transformed[:,1], c = "blue", marker = ".")

    plt.draw()
def isomap_3d(df, n_neighbours):
    # Scale section
    df = StandardScale(df)

    # Isomap section
    iso = manifold.Isomap(n_neighbours, n_components = 3)
    iso.fit(df)

    transformed = iso.transform(df)

    # Visualization
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_title("Isomap 3D")

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