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


def extract(df, idx, share_volume, budget):
    'TO DO: YOU HAVE TO COME BACK TO THIS FUCTION AND MAKE IT CORRECT'
    datetime = pd.to_datetime(df.iloc[idx][18])
    price = df.iloc[idx][1]
    name = df.iloc[idx][0]
    target = df[df['name'] == name]
    future_price = []
    for target_date, obs in target.groupby(by=target['datetime'].values.astype('<M8[D]')):
      deltatime = pd.Timedelta(pd.to_datetime(obs.iloc[0]['datetime']).date() - datetime.date()).days
      if 0 < deltatime and 20 > deltatime:
        a = (((obs['price'].mean() - price) / 100 + (obs['ydp'].mean() - price) / 100) /2) * share_volume * price
        # print('consider this ', deltatime, a, price, obs['price'].mean(), share_volume, budget, round(a/budget, 4), (price*share_volume)/budget)
        future_price.append([deltatime, round(a/budget, 5), round(a, 5)])

    return pd.DataFrame(future_price, columns=['deltatime', 'percent', 'asset']) #[delta day, mean() all the day price]


def check_portfolio(df, datetime, portfolio, budget):
    done = False
    wealth = []
    datetime = pd.to_datetime(datetime)
    for i in portfolio:
        name = i[0]
        target = df[df['name'] == name]
        for target_date, obs in target.groupby(by=target['datetime'].values.astype('<M8[D]')):
            if target_date == datetime.date():
                wealth.append(obs['price'].mean() * i[1])

    a = sum(wealth) + budget
    print(a)
    if a < 16000000:
        done = True
    return done

    # for target_date, obs in df.groupby(by=df['datetime'].values.astype('<M8[D]')):


    # df['datetime'] = pd.to_datetime(df['datetime'])
    # # print(df['datetime'].dt.year)
    # # print(df['datetime'].dt )
    # dt0 = df[df['datetime'].dt.year == cond.year]
    # # print(dt0)
    # dt1 = dt0[dt0['datetime'].dt.month == cond.month]
    # dt2 = dt1[dt1['datetime'].dt.day == cond.day]
    # print(dt2)
    # for i in portfolio:

