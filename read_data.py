import numpy as np
import pandas as pd

# pd.set_option('use_inf_as_na', True)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 500)
address = '/Users/amir/Desktop/data/'
file_postfix = '/read.xlsx'

colums = ['datetime', 'name', 'price', 'ydp', 'wp_fi',
          'change', 'change_per', 'd_high', 'd_low',
          'eps', 'pe', 'trade_volume', 'trade_value',
          'trade_amount', 'tepix', 'dollar', 'purchase_q',
          'sale_q', 'condition']


# trade_volume                      4.74504e-05
# trade_value                       0.000661827
# trade_amount                      0.000441521

# purchase_q                        6.06755e-06
# sale_q                            4.67584e-05


class dataset:

  def __init__(self):

    self.df = self.read()
    # self.df =
    self.sname = self.names_list()
    self.norm_df = self.normalize()
    # self.df = self.remove_(a=)

    # self.df.reset_index(drop=True)

  def read(self, write=False):

    if write:
      data = pd.read_excel(address + '1' + file_postfix)

      for i in range(2, 3):
        print(str(i))
        temp = pd.read_excel(address + str(i) + file_postfix)
        data = data.append(temp, ignore_index=True)
        data = data.sort_values(by="datetime")
        data.to_excel('Data.xlsx')

    else:
      data = pd.read_excel('Data.xlsx')
      data = self.remove_(data , ['جم پیلن', 'جم'])
      data = self.log2(data, ['trade_volume', 'trade_value', 'trade_amount', 'purchase_q', 'sale_q'])

    return data

  def normalize(self):
    dt = self.read()
    numeric_cols = dt.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
      if 'Unnamed' not in col:
        dt[col] = (dt[col] - dt[col].mean()) / dt[col].std()
        dt[col] = (dt[col] - dt[col].min()) / (dt[col].max() - dt[col].min())

    # df.set_index(['datetime', 'name'], inplace=True)
    return dt


  def names_list(self):

    list_ = self.df.drop_duplicates(subset="name", keep='first')
    a = list_['name'].reset_index(drop=True)
    a = a.map(lambda x: x.replace('هلدينگ', ''))
    a = a.map(lambda x: x.replace(')', ''))
    a = a.map(lambda x: x.replace('(', ''))

    return a.drop_duplicates(keep='first')

  def log2(self, data, list_):
    for x in list_:
      data[x] = np.log2(data[x])
      data[x] = data[x].replace(-np.inf, 0)
      data[x] = data[x].replace(np.inf, 0)
    return data

  def remove_(self, df, list_):
    for i in list_:
      temp = df[df.name != i]
    return temp.reset_index(drop=True)



# data = dataset()




# data = data.remove_(a='جم')
# data = data.remove_(a='جم پیلن')

# print(data.sname)


# df = normalize(data)
# df = remove_(df, 'جم')
# df = remove_(df, 'جم پیلن')
# list_name = names_list(df)
#
# list_name.to_excel('one.xlsx')
# for i, row in a.items():
#     # print(row)
#     print(i)
#     if "هلدينگ (" in row:
#         v = row.replace("هلدينگ (", '')
#         print(v)
#         a.iloc[i] = v

# x = jj.iloc[:, 1].values
# print(x[0:4])
# le = LabelEncoder()
# y = le.fit_transform(x)
# ohe = OneHotEncoder()
# z = ohe.fit_transform(y).toarray()
# print(x, y, z)
# print(x.shape, y.shape, z.shape)
