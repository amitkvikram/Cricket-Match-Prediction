import numpy as np
import pandas as pd
import os, zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm, tqdm_notebook
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

data_dir_name = "./data"
zip_dir_name = "./data/t20s_male.zip"
zip_ref = zipfile.ZipFile(zip_dir_name, "r")
filenames = list(map(lambda x:x.filename, zip_ref.filelist))

team_match_cnt = {}
#Create a dataframe with (filaname, date)
file_info_list=[]
for filename in tqdm_notebook(filenames):
    if filename.endswith(".yaml"):
        with zip_ref.open(filename, 'r') as stream:
            data_loaded = yaml.load(stream)
            #exclude draw matches and D/L
            if 'result' in list(data_loaded['info']['outcome'].keys()) or 'method' in list(data_loaded['info']['outcome'].keys()):
                continue
            date = data_loaded['info']['dates'][0]

            teamBatting = data_loaded['innings'][1]['2nd innings']['team']
            teamBowling = data_loaded['innings'][0]['1st innings']['team']
            if teamBatting not in team_match_cnt:
                team_match_cnt[teamBatting] = 0
            if teamBowling not in team_match_cnt:
                team_match_cnt[teamBowling] = 0
            team_match_cnt[teamBatting] += 1
            team_match_cnt[teamBowling] += 1

            row = (filename, date, teamBatting, teamBowling)
            file_info_list.append(row)
file_info_list = [dict(zip(['filename','date'], [filename, date])) for filename,date,teamBatting,teamBowling in file_info_list 
                if (team_match_cnt[teamBatting]>5 and team_match_cnt[teamBowling] > 5)]
file_df = pd.DataFrame(file_info_list)
file_df['date'] = pd.to_datetime(file_df['date'])
file_df.sort_values(by=['date'], inplace=True)

train_file_df, val_file_df =  train_test_split(file_df, test_size=0.2, shuffle=False)
val_file_df, test_file_df = train_test_split(val_file_df, test_size=0.5, shuffle=False)


cols = ['team-batting', 'team-bowling', 'wicket-left', 'runs-scored', 'target-score', 'balls-bowled',
        'batsman-score', 'nonstriker-score', 'batsman-balls-faced', 'nonstriker-balls-faced',
        'match-id', 'winner', 'date']

def getRow(data_loaded, matchId, filename):
    match_data = []

    date = pd.to_datetime(data_loaded['info']['dates'][0])
    targetScore, runsScored, wicketLeft = 0, 0, 10
    ballsBowled = 0
    playerRuns = {}
    teamBatting = data_loaded['innings'][1]['2nd innings']['team']
    teamBowling = data_loaded['innings'][0]['1st innings']['team']
    for x in data_loaded['innings'][0]['1st innings']['deliveries']:
        targetScore += x[list(x.keys())[0]]['runs']['total']
    winner = data_loaded['info']['outcome']['winner']
    push = True

    for x in (data_loaded['innings'][1]['2nd innings']['deliveries']):
        #Update Batsman Score
        key = list(x.keys())[0]
        striker = x[key]['batsman']
        nonStriker = x[key]['non_striker']
        if(striker not in playerRuns):
            playerRuns[striker] = [0, 0]     #[ballsPlayed, runsScore]
        if(nonStriker not in playerRuns):
            playerRuns[nonStriker] = [0, 0]

        values = [teamBatting, teamBowling, wicketLeft, runsScored, targetScore, ballsBowled, 
                   playerRuns[striker][1], playerRuns[nonStriker][1], 
                   playerRuns[striker][0], playerRuns[nonStriker][0],
                   matchId, winner, date]
        if push: match_data.append(dict(zip(cols, values)))
        else: match_data[-1] = dict(zip(cols, values))

        wicketLeft -= int('wicket' in x[key])
        runsScored += x[key]['runs']['total']
        playerRuns[striker][1] += x[key]['runs']['batsman']
        
        #If this ball is not wide or no ball then increment ballsPlayed
        if 'extras' not in x[key]:
            playerRuns[striker][0] += 1
            ballsBowled += 1
            push = True
        elif 'legbyes' in x[key]['extras'] or 'byes' in x[key]['extras']:
            ballsBowled += 1
            push = True
        else: push = False
        
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