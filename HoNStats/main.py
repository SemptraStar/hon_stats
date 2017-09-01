import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

from bs4 import BeautifulSoup

import dataloader as dload
import visualization as vs
import clustering
import transformation
import classification


data_path = os.getcwd() + "\\Data\\"

players = dload.getStatsCsv().drop(["WIN RATE", "WINS", "LOSSES"], axis=1)

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

plt.show()
"""

print(dload.hero_roles)

carry = ["`M1ndGames", "Mr`FF", "Imbaboy", "baltazar`", "cHHHI"]
mid = ["B1zzyP", "Mewtu`", "Amenjagemid`", "Xgodxd", "JustMolly"]
jungle = ["m`JEEY", "Dzili", "FairyHot", "Wutwoot", "WhaT_YoU_GoT"]
hard = ["DopeKlD", "Polymorpha`", "BilboSwagin`", "Cptain`Ahue", "lzi`"]
support = ["`Chewy", "BabyMuffin", "nichter90", "claudianus", "`VanG"]

carry = players.loc[players['PLAYER'].isin(carry)]
mid = players.loc[players['PLAYER'].isin(mid)]
jungle = players.loc[players['PLAYER'].isin(jungle)]
hard = players.loc[players['PLAYER'].isin(hard)]
support = players.loc[players['PLAYER'].isin(support)]

samples = pd.concat([carry, mid, jungle, hard, support]).drop(["GAME LENGTH", "RANK", "PLAYER"], axis=1)

roles = ["Mid"] * 5 + \
        ["Carry"] * 5 + \
        ["Jungle"] * 5 + \
        ["Hard"] * 5 + \
        ["Support"] * 5

model = classification.kneighbors(samples, roles, 3)

p = players.drop(["GAME LENGTH", "RANK", "PLAYER"], axis=1)
print(players.drop(["GAME LENGTH"], axis=1).to_string())
classes = []
for i in range(p.shape[0]):
    class_ = model.predict(p.iloc[i,:].values.reshape(1, -1))
    #print("{0}: {1} - {2}".format(i, players.iloc[i,:]["PLAYER"], class_))
    classes.extend(class_)

print(classes)

players["ROLE"] = classes
print(players)

colors = pd.read_html("http://www.rapidtables.com/web/color/RGB_Color.htm")[3].loc[3:, "Hex Code#RRGGBB"].reset_index(drop = True)
vs.classes_plot_2d(players.drop(["GAME LENGTH", "RANK", "PLAYER"], axis=1), "ROLE", \
   ["Mid", "Carry", "Jungle", "Hard", "Support"], colors)
vs.classes_plot_2d(players.drop(["GAME LENGTH", "RANK", "PLAYER"], axis=1), "ROLE", \
   ["Mid", "Carry", "Jungle", "Hard", "Support"], colors, transform="iso")

plt.show()