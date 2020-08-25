# import required libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot(action):

  act_df = pd.DataFrame(action, columns=['actions'])
  act_df['actions_type'] = act_df['actions'].astype('category')
  act_df['actions_type_num'] = act_df['actions_type'].cat.codes

  # creating instance of one-hot-encoder
  enc = OneHotEncoder(handle_unknown='ignore')
  enc_df = pd.DataFrame(enc.fit_transform(act_df[['actions_type_num']]).toarray())
  act_df = act_df.join(enc_df).drop(columns=['actions_type'])
  act_df = act_df.set_index(['actions_type_num', 'actions'])

  return act_df.sort_index()

def detect_action(a):

  action = ('Buy', 'Sell', 'Pass')
  t = one_hot(action)
  a = pd.Series(a).argmax()

  return t.loc[a].values[0], t.loc[a].index.values


def extract(df, idx):

  datetime = pd.to_datetime(df.iloc[idx][18])
  name = df.iloc[idx][0]
  target = df[df['name'] == name]
  future_price = []
  for target_date, obs in target.groupby(by=target['datetime'].values.astype('<M8[D]')):
    deltatime = pd.Timedelta(pd.to_datetime(obs.iloc[0]['datetime']).date() - datetime.date()).days
    if 0 < deltatime:
      if 20 > deltatime:
        future_price.append([deltatime, obs['price'].mean()])

  return future_price  # [delta day, mean() all the day price ]



# for i in range(len(target)):
#   deltatime = pd.Timedelta(target.iloc[i]['datetime'] - datetime)
