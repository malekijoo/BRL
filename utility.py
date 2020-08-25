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

  datetime = df.iloc[idx][18]
  name = df.iloc[idx][0]
  target = df[df['name'] == name]
  print(target)
  # print(target.groupby(by=['datetime']).groups)
  target = target.groupby(by=target['datetime'].values.astype('<M8[D]')).groups
  # print(target.groupby(by=target['datetime'].dt.date).count())


  # print(target)
  lis_rw_base_on_day = []

  # for i in range(len(target)):
  #   deltatime = pd.Timedelta(target.iloc[i]['datetime'] - datetime)
    # print(deltatime.value)
    # if deltatime.min > 0:
    #   print(i, deltatime, i)
    # lis_rw_base_on_day[deltatime][j].append([target.iloc[i]])
    # j += 1
    # print(lis_rw_base_on_day)

  # print(lis_rw_base_on_day)
  # print(type(target.iloc[1]['datetime']))


  # print(pd.to_datetime(target.iloc[1]['datetime']).year)
  # target['pd_datetime'] = pd.to_datetime(target['datetime'])
  # print(type(pd.to_datetime(target.iloc[1]['datetime'])))
  # print(type(target.iloc[1]['pd_datetime']))


  # b = t + pd.Series(a)
  # print(b)
  # # for i in range(len(t)):
  # print(t.idxmax(axis=0))


    # b, ind = a.index(max(t[i, :] + a)), i

  # print(b, ind)

# action = ('Buy', 'Sell', 'Pass')
# t = one_hot(action)
# print(t)
# a, action = detect_action([0.583, 0.233, 0.184])
# print(a, action)
