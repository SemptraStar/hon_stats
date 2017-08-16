import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from bs4 import BeautifulSoup
import urllib
import re
import json

def getStatsHttp():
    connection_string = "http://client.sea.heroesofnewerth.com/index.php?r=site/rankladder&ranktype=normal&hongameclientcookie=|naeu|&page="
    stats = []

    for i in range(1, 41):
        page = connection_string + str(i)
        stats.append(pd.read_html(page)[0])

    return pd.concat(stats)
def getStatsCsv():
    file = os.getcwd() + "\\Data\\players_stats.csv"
    return pd.read_csv(file, index_col = 0)

def saveStatsCsv(stats):
    stats.to_csv(os.getcwd() + "\\Data\\players_stats.csv")

def getPlayerHeroesNums(nickname):
    page = ""   
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
    req = urllib.request.Request("https://www.heroesofnewerth.com/playerstats/ranked/" + nickname, headers = hdr)

    with urllib.request.urlopen(req) as response:
        page = str(response.read())

    soup_page = BeautifulSoup(''.join(page), "lxml")
    heroes_divs = soup_page.find_all('div', {"class" : "iconHolder regular white fontXS"})

    heroes = []

    for div in heroes_divs:
        hero = re.search("/[0-9]+/", str(div)).group(0)
        heroes.append(hero[1:-1])

    return heroes
def getPlayerHeroes(allHeroes, nums):
    heroes = []

    for num in nums:
        heroes.append(allHeroes[num])

    return heroes
def getPlayerFavHeroes(nickname):
    file = os.getcwd() + "\\Data\\players_fav_heroes.json"
    with open(file, "r") as fav_heroes_list:
        heroes = json.load(fav_heroes_list)
        if nickname in heroes:
            return heroes[nickname]
        else:
            return []   

def getAllHeroesJson(heroesAsKeys = False):
    file = os.getcwd() + "\\Data\\all_heroes.json"
    with open(file, "r") as heroes_list:
        list = json.load(heroes_list)

        heroes = list[0]
        nums = list[1]

        if heroesAsKeys:
            return dict(zip(heroes, nums))
        else:
            return dict(zip(nums, heroes))
def getAllHeroesHttps(url, heroesAsKeys = False):
    all_heroes_page = getHttpsPage(url)
    soup_heroes = BeautifulSoup(''.join(all_heroes_page), "lxml")
    heroes_divs = soup_heroes.find_all('div', {"class" : "over default"})

    heroes = []

    for div in heroes_divs:
        div = str(div)
        hero = div[div.find(">") + 1 : div.rfind("<")]
        heroes.append(hero)

    hero_nums = soup_heroes.find_all('div', {"class" : "heroIcon"})

    nums = []

    for div in hero_nums:
        div = str(div)
        hero = re.search("/[0-9]+/",div)
        if hero:
            nums.append(hero.group(0)[1:-1])

    if heroesAsKeys:
        return dict(zip(heroes, nums))
    else:
        return dict(zip(nums, heroes))

def getHeroesRoles():
    """
    heroes = {"Andromeda" : "Support", "Artillery" : "Carry", "Blitz" : "Support",
              "Emerald Warden" : "Carry", "Engineer" : "Support", "Magebane" : "Carry",
              "Master of Arms" : "Carry", "Moira" : "Support", "Monkey King" : "Mid",
              "Moon Queen" : "Carry", "Night Hound" : "Carry", "Nitro" : "Carry",
              "Nomad" : "Mid", "Scout" : "Carry", "Silhouette" : "Carry",
              "Sir Benzington" : "Mid", "Swiftblade" : "Carry", "Tarot" : "Carry",
              "Valkyrie" : "Hard", "Wildsoul" : "Jungle", "Zephyr" : "Jungle",
              "Adrenaline" : "Carry", "Arachna" : "Mid", "Blood Hunter" : "Mid",
              "Bushwack" : "Carry", "Calamity" : "Mid", "Chronos" : "Carry",
              "Corrupted Disciple" : "Carry", "Dampeer" : "Mid", "Fayde" : "Mid",
              "Flint Beastwood" : "Mid", "Forsaken Archer" : "Carry", "Gemini" : "Carry",
              "Grinex" : "Hard", "Gunblade" : "Mid", "Klanx" : "Carry",
              "Riptipe" : "Mid", "Sand Wraith" : "Carry", "Shadowblade" : "Carry",
              "Slither" : "Support", "Soulstealer" : "Mid", "The Dark Lady" : "Carry",
              "The Madman" : "Carry", "Tremble" : "Carry", "Aluna" : "Support",
              "Blacksmith" : "Support", "Bombardier" : "Mid", "Bubbles" : "Mid",
              "Ellonia" : "Support", "Empath" : "Support", "Kinesis" : "Support",
              "Martyr" : "Support", "Monarch" : "Support", "Nymphora" : "Support",
              "Oogie" : "Carry", "Ophelia" : "Jungle", "Pearl" : "Support",
              "Pollywog Priest" : "Mid", "Pyromancer" : "Mid", "Rhapsody" : "Support",
              "Skrap" : "Support", "Tempest" : "Jungle", "The Chipper" : "Mid",
              "Thunderbringer" : "Mid", "Vindicator" : "Support", "Warchief" : "Mid",
              "Witch Slayer" : "Support", "Artesia" : "Mid", "Circe" : "Support",
              "Defiler" : "Mid", "Demented Shaman" : "Support", "Doctor Repulsor" : "Mid",
              "Geomancer" : "Hard", "Glacius" : "Support", "Gravekeeper" : "Support",
              "Hellbringer" : "Support", "Myrmidon" : "Support", "Parallax" : "Mid",
              "Parasite" : "Jungle", "Plague Rider" : "Hard", "Prophet" : "Support",
              "Puppet Master" : "Carry", "Revenant" : "Support", "Riftwalker" : "Support",
              "Soul Reaper" : "Mid", "Succubus" : "Support", "Torturer" : "Mid",
              "Voodoo Jester" : "Support", "Wretched Hag" : "Mid", "Armadon" : "Hard",
              "Behemoth" : "Hard", "Berzerker" : "Carry", "Bramble" : "Hard", 
              "Drunken Master" : "Mid", "Flux" : "Hard", "Hammerstorm" : "Carry",
              "Ichor" : "Support", "Jeraziah" : "Support", "Keeper of the Forest" : "Jungle",
              "Legionnaire" : "Jungle", "Midas" : "Hard", "Pandamonium" : "Mid",
              "Pebbles" : "Mid", "Predator" : "Carry", "Prisoner 945" : "Mid",
              "Rally" : "Hard", "Rampage" : "Hard", "Salomon" : "Carry",
              "Shellshock" : "Support", "Solstice" : "Jungle", "The Gladiator" : "Mid",
              "Tundra" : "Hard", "Accursed" : "Support", "Amun-Ra" : "Jungle",
              "Apex" : "Carry", "Balphagore" : "Hard", "Cthulhuphant" : "Jungle",
              "Deadlift" : "Hard", "Deadwood" : "Mid", "Devourer" : "Mid",
              "Draconis" : "Jungle", "Electrician" : "Hard", "Gauntlet" : "Mid",
              "Kane" : "Hard", "King Klout" : "Mid", "Kraken" : "Hard",
              "Lodestone" : "Hard", "Lord Salforis" : "Mid", "Magmus" : "Hard",
              "Maliken" : "Carry", "Moraxus" : "Hard", "Pestilence" : "Hard",
              "Pharaoh" : "Hard", "Ravenor" : "Carry", "War Beast" : "Jungle" }
    """

    file = "/Data/heroes_roles.json"
    with open(file, "r") as roles:
       list = json.load(roles)
       return list
def describeRoles(heroes, allRoles):
    roles = []

    for hero in heroes:
        if hero in allRoles:
            roles.append(allRoles[hero])

    return roles
        
def getHttpsPage(url): 
     hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
     req = urllib.request.Request(url, headers = hdr)

     with urllib.request.urlopen(req) as response:
         return str(response.read())

def getPlayerDetailedStats(nickname):
    page = getHttpsPage("https://www.heroesofnewerth.com/playerstats/ranked/" + str(nickname))
    soup_page = BeautifulSoup(''.join(page), "lxml")

    stats = soup_page.find_all('div', {'class' : 'column right regular greyLight fontXS'})

    if len(stats) == 0:
        return []

    indexes = str(stats[1])
    s = re.findall(r"\\t\\t\\t\\t(.*?)<", indexes)

    return [s[0], s[2], s[7], s[8]]


ranks = ["Bronze III", "Bronze II", "Bronze I",
        "Silver III", "Silver II", "Silver I",
        "Gold III", "Gold II", "Gold I",
        "Diamond III", "Diamond II", "Diamond I",
        "Legendary", "Immortal"]