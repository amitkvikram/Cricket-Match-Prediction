import numpy as np
import pandas as pd
import os, zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm, tqdm_notebook
# from yaml import CLoader as loader
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

data_dir_name = "./data"
zip_dir_name = "./data/t20s_male.zip"
zip_ref = zipfile.ZipFile(zip_dir_name, "r")
filenames = list(map(lambda x:x.filename, zip_ref.filelist))

#Create a dataframe with (filaname, date)
file_info_list=[]
for filename in tqdm_notebook(filenames):
    if filename.endswith(".yaml"):
        with zip_ref.open(filename, 'r') as stream:
            data_loaded = yaml.load(stream)
            #exclud draw matches
            if 'result' in list(data_loaded['info']['outcome'].keys()):
                continue
            date = data_loaded['info']['dates'][0]
            row = dict(zip(['filename','date'], [filename, date]))
            file_info_list.append(row)
file_df = pd.DataFrame(file_info_list)
file_df['date'] = pd.to_datetime(file_df['date'])
file_df.sort_values(by=['date'], inplace=True)

train_file_df, val_file_df =  train_test_split(file_df, test_size=0.2, shuffle=False)
val_file_df, test_file_df = train_test_split(val_file_df, test_size=0.5, shuffle=False)

cols = ['wicket-left', 'runs-scored', 'team-batting', 'team-bowling', 'target-score',
        'batsman-score', 'nonstriker-score', 'batsman-balls-faced', 'nostriker-balls-faced',
        'match-id', 'balls-bowled', 'this-ball-wicket', 'this-ball-run', 'winner', 'date']

def getRow(data_loaded, matchId, filename):
    match_data = []

    date = pd.to_datetime(data_loaded['info']['dates'][0])
    targetScore, runsScored, wicketLeft = 0, 0, 10
    thisBallWicket, thisBallRun = 0, 0
    ballsBowled = 0
    playerRuns = {}
    teamBatting = data_loaded['innings'][1]['2nd innings']['team']
    teamBowling = data_loaded['innings'][0]['1st innings']['team']
    for x in data_loaded['innings'][0]['1st innings']['deliveries']:
        targetScore += x[list(x.keys())[0]]['runs']['total']
    winner = data_loaded['info']['outcome']['winner']

    for x in (data_loaded['innings'][1]['2nd innings']['deliveries']):
        #Update Batsman Score
        striker = x[list(x.keys())[0]]['batsman']
        nonStriker = x[list(x.keys())[0]]['non_striker']
        if(striker not in playerRuns):
            playerRuns[striker] = [0, 0]     #[ballsPlayed, runsScore]
        if(nonStriker not in playerRuns):
            playerRuns[nonStriker] = [0, 0]


        wicketLeft -= int('wicket' in x[list(x.keys())[0]])
        thisBallWicket = int('wicket' in x[list(x.keys())[0]])
        runsScored += x[list(x.keys())[0]]['runs']['total']
        thisBallRun = x[list(x.keys())[0]]['runs']['total']
        playerRuns[striker][1] += x[list(x.keys())[0]]['runs']['batsman']
        
        #If this ball is not wide or no ball then increment ballsPlayed
        if 'extras' not in x[list(x.keys())[0]]:
            playerRuns[striker][0] += 1
            ballsBowled += 1
        elif 'legbyes' in x[list(x.keys())[0]]['extras'] or 'byes' in x[list(x.keys())[0]]['extras']:
            ballsBowled += 1

        values = [wicketLeft, runsScored, teamBatting, teamBowling, targetScore,
                   playerRuns[striker][1], playerRuns[nonStriker][1], 
                   playerRuns[striker][0], playerRuns[nonStriker][0],
                   matchId, ballsBowled, thisBallWicket, thisBallRun, winner, date]

        match_data.append(dict(zip(cols, values)))
        
    return match_data

# Get train, validation and test dataframes
matchId = 0
train_data, val_data, test_data = [], [], []
for filename in tqdm(train_file_df.loc[:, 'filename']):
    with zip_ref.open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)
        match_data = getRow(data_loaded, matchId, filename)
        matchId+=1 
        stream, data_loaded = None, None
        train_data += match_data

for filename in tqdm(val_file_df.loc[:, 'filename']):
    with zip_ref.open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)
        match_data = getRow(data_loaded, matchId, filename)
        matchId+=1
        stream, data_loaded = None, None
        val_data += match_data

for filename in tqdm(test_file_df.loc[:, 'filename']):
    with zip_ref.open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)
        match_data = getRow(data_loaded, matchId, filename)
        matchId+=1
        stream, data_loaded = None, None
        test_data += match_data

pd.DataFrame(train_data).to_csv("./data/train_data.csv")
pd.DataFrame(val_data).to_csv("./data/val_data.csv")
pd.DataFrame(test_data).to_csv("./data/test_data.csv")