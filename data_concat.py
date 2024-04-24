'''
Author: Chi Zhang 944376823@qq.com
Date: 2023-07-10 14:42:55
LastEditors: Chi Zhang 944376823@qq.com
LastEditTime: 2023-07-12 18:03:35
FilePath: \\Newsvendor-related\\data_concat.py
Description: 
Copyright (c) 2023 by Zhang Chi, All Rights Reserved. 
'''
import pandas as pd
import os


account_pre = pd.read_csv(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20221101-20230420\account.csv')
# 按busdate和code排序
account_pre.sort_values(by=['busdate', 'code'], inplace=True)

account_post = pd.read_csv(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20230421-20230701\account.csv')
account_post.sort_values(by=['busdate', 'code'], inplace=True)

if account_pre['busdate'].max() >= account_post['busdate'].min():
    print('account_pre和account_post的日期有重叠')
    exit()
else:
    account = pd.concat([account_pre, account_post], axis=0)

# 若HLJ销售数据20221101-20230701文件夹不存在，则创建
if not os.path.exists(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20221101-20230701'):
    os.mkdir(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20221101-20230701')

# 保存account
account.to_csv(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20221101-20230701\account.csv', index=False)


running_pre = pd.read_csv(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20221101-20230420\running.csv')
running_pre.sort_values(by=['selldate', 'selltime', 'code'], inplace=True)

running_post = pd.read_csv(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20230421-20230701\running.csv')
running_post.sort_values(by=['selldate', 'selltime', 'code'], inplace=True)

if running_pre['selldate'].max() >= running_post['selldate'].min():
    print('running_pre和running_post的日期有重叠')
    exit()
else:
    running = pd.concat([running_pre, running_post], axis=0)

# 保存running
running.to_csv(r'd:\Work info\WestUnion\data\origin\HLJ销售数据20221101-20230701\running.csv', index=False)

print('data_concat.py 运行完毕！')