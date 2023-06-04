import pandas as pd

# Read the data
data = pd.read_csv(r"d:\Work info\WestUnion\data\origin\HLJ\account.csv")
# busdate排序
data = data.sort_values(by='busdate', ascending=True)

data_sec = pd.read_csv(r"d:\Work info\WestUnion\data\origin\HLJ销售数据20221101-20230420\account.csv")
data_sec = data_sec.sort_values(by='busdate', ascending=True)
