import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import json
import os, sys
import dataloader as dload
import visualization as vsz

data_path = os.getcwd() + "\\Data\\"

players = dload.getStatsCsv()

print(players)
print(list(players))

players = players.drop(["PLAYER", "GAME LENGTH"], 1)


vsz.andrews_curves(players, "RANK", False)
vsz.andrews_curves(players, "RANK", True)

vsz.draw_all()
