import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import json
import os, sys
import dataloader as dload
import visualization as vs
import clustering
import transformation

data_path = os.getcwd() + "\\Data\\"

players = dload.getStatsCsv()

print(players)

# Visualization
"""
players = players.drop(["PLAYER", "GAME LENGTH"], 1)

vs.parallel_coords(players, "RANK", normalize = True)
vs.andrews_curves(players, "RANK", normalize = True)

players = players.drop(["RANK"], 1)

vs.pca_2d(players, normalize = True)
vs.pca_3d(players, normalize = True)
vs.isomap_2d(players, 6)
vs.isomap_3d(players, 6)

vs.draw_all()
"""

# !!!
# Results are pretty the same
# !!!

colors = pd.read_html("http://www.rapidtables.com/web/color/RGB_Color.htm")[3].loc[3:, "Hex Code#RRGGBB"].reset_index(drop = True)
gx = players[players.columns.difference(["PLAYER", "GAME LENGTH", "WIN RATE", "RANK", "WINS", "LOSSES"])]
players_iso = pd.DataFrame(transformation.isomap(gx, 3, 4))
print(players_iso)

groups = int(input("Enter number of clusters: "))
clusters = clustering.kmeans(players_iso, groups)

for g in range(groups):
    gr = players[clusters.labels_ == g].reset_index(drop = True)
    print()
    print("Group #" + str(g))
    print(gr.head(10))
    print(gr.describe().loc["mean", :])

players["group"] = clusters.labels_
players = players.iloc[:100, :]
vs.parallel_coords(players.drop(["PLAYER", "GAME LENGTH"], 1), "group", True)

vs.cluster_scatter_3d(players_iso, clusters, colors)

# !!!
# Good results
# !!!

gx = players.loc[:, ["WARDS", "A", "GPM"]]
groups = int(input("Enter number of clusters: "))
clusters = clustering.kmeans(gx, groups, True)

for g in range(groups):
    gr = players[clusters.labels_ == g].reset_index(drop = True)
    print()
    print("Group #" + str(g))
    print(gr.head(10))
    print(gr.describe().loc["mean", :])


vs.cluster_scatter_3d(gx, clusters, colors, xlabel = "WARDS", ylabel = "D", zlabel = "GPM")

players["group"] = clusters.labels_
players = players.iloc[:100, :]
vs.parallel_coords(players.drop(["PLAYER", "GAME LENGTH"], 1), "group", True)


plt.show()