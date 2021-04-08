import pandas as pd 
from bs4 import BeautifulSoup 
import requests
import string
import re


def get_lyrics(url):
    '''
    scrape lyrics for a song from genius.com
    Input:
    url (str): the genius link for a song

    Output:
    lyrics (a list of strs): tokenized lyrics of the song requested 
    '''
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    soup.prettify()
    verses = soup.find_all("div", class_ ="lyrics")
    lyrics = ""
    for verse in verses:
        lyrics = lyrics + verse.text.lower()
    lyrics = lyrics.replace("\n", " ")
    lyrics = re.sub(r"\[.+?\]", "", lyrics)
    lyrics = re.sub(r"\(.+?\)", "", lyrics).lstrip().rstrip()
    lyrics = re.sub("\u2005", " ", lyrics)
    #lyrics = lyrics.translate(str.maketrans("", "", string.punctuation))
    return lyrics

def find_top_artists(year, female = True):
    '''
    find top 10 artists according to Billbroad Year-End Char
    Input:
    year (str): year of interest (2006-2020)
    female (bool): whether the chart is for female or male artists 
    Output:
    top_artists (list of strs): a list of top 10 female/male artists
    '''
    if female == True:
        r = requests.get("https://www.billboard.com/charts/year-end/{}/top-artists-female".format(year))
    else:
        r = requests.get("https://www.billboard.com/charts/year-end/{}/top-artists-male".format(year))
    soup = BeautifulSoup(r.text, "html.parser")
    top_artists = [artist.text.strip() for artist in soup.find_all("div", class_="ye-chart-item__title")]
    return top_artists


def get_songs_links(artist):
    '''
    get 10 top songs of an artist
    Input:
    artist (str): the name of the artist
    Output:
    songs (list)L a list of strings of song links
    '''
    songs = []
    artist_query = re.sub(" ", "-", artist)
    url = "https://genius.com/artists/{}".format(artist_query)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    songs =[song.get("href") for song in soup.find_all("a", class_= "mini_card")]
    return songs

