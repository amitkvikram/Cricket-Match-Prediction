import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

class PreprocessingHelper:
    feature_cols = ['team-batting', 'team-bowling', 'wicket-left', 'runs-scored', 'target-score', 'balls-bowled',
                'batsman-score', 'nonstriker-score', 'batsman-balls-faced', 'nonstriker-balls-faced']
    max_sequence_length = 120

    def __init__(self):
        train_df = pd.read_csv("./data/train_data.csv", index_col = 0)
        val_df = pd.read_csv("./data/val_data.csv", index_col = 0)
        test_df = pd.read_csv("./data/test_data.csv", index_col = 0)

        self.le = LabelEncoder()
        x = train_df['team-batting'].tolist() + train_df['team-bowling'].tolist()
        x = x + val_df['team-batting'].tolist() + val_df['team-bowling'].tolist()
        x = x + test_df['team-batting'].tolist() + test_df['team-bowling'].tolist()
        self.le.fit(x)

        train_df = self.encodedLabels(train_df)
        train_df['balls-left'] = 120 - train_df['balls-bowled']
        # train_df['runs-req'] = train_df['target-score'] - train_df['runs-scored']
        train_df = train_df.loc[train_df['balls-left']>0, :]
        self.std = np.std(train_df.loc[:, self.feature_cols[2:]], axis = 0)
        self.mean = np.mean(train_df.loc[:, self.feature_cols[2:]], axis = 0)

    def getLabelEncoding(self, x):
        return self.le.transform(x) + 1

    def encodedLabels(self, df):
        df.loc[:, 'team-batting'] = self.getLabelEncoding(df['team-batting'])
        df.loc[:, 'team-bowling'] = self.getLabelEncoding(df['team-bowling'])
        df.loc[:, 'winner'] = self.getLabelEncoding(df['winner'])

        return df

    def normalize(self, df):
        df.loc[:, self.feature_cols[2:]] = (df.loc[:, self.feature_cols[2:]] - self.mean) / self.std
        return df

    def getPredictionLabel(self, df):
        boolmap = (df.winner == df['team-batting'])
        df.loc[:, 'prediction'] = np.array(boolmap).astype(np.float)
        return df

    def getX_Y(self, df):
        n, m = df.agg({'match-id':'nunique'})[0], len(self.feature_cols)
        X, Y = np.zeros((n, self.max_sequence_length, m)), np.zeros((n, self.max_sequence_length, 1))
        grouped = df.groupby(['match-id'])
        i = 0
        for group_name, group in tqdm(grouped):
            j = 0
            for row_index, row in group.iterrows():
                if j<self.max_sequence_length:
                    Y[i, j, 0] = row['prediction']
                    X[i, j, :] = row[self.feature_cols].values
                j = j + 1
            i = i + 1
        return X, Y
    
    def preprocess(self, df):
        df.loc[:, 'balls-left'] = 120 - df['balls-bowled']
        # df['runs-req'] = df['target-score'] - df['runs-scored']
        df = df.drop(df.loc[df['balls-left']<=0].index)
        df = self.getPredictionLabel(df)
        df_c = df.copy()
        df = self.encodedLabels(df)
        df = self.normalize(df)

        return df, df_c