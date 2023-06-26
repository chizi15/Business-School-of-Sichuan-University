# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 6)


input_path = r'D:\Work info\WestUnion\data\processed\HLJ\脱敏及筛选后样本数据\output'
output_path = r"D:\Work info\SCU\MathModeling\2023\data\output"
output_path_self_use = r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata"
first_day = '2020-07-01'
last_day = '2023-04-20'
sm_sort_name = ['食用菌', '花叶类', '水生根茎类', '辣椒类', '茄类', '花菜类']
unit_cost_critical = 0  # 进货单价的筛选阈值，小于等于该值的数据将被剔除


commodity = pd.read_csv(f'{input_path}/commodity.csv')
# 先转成int64，以免位数超限被转换为负数
if not isinstance(commodity['code'].iloc[0], str):
    commodity[['code', 'sm_sort', 'md_sort', 'bg_sort']] = commodity[['code', 'sm_sort', 'md_sort', 'bg_sort']].astype('Int64').astype(str)
# commodity按sm_sort_name进行第一次筛选
commodity = commodity[commodity['sm_sort_name'].isin(sm_sort_name)]
commodity = commodity[~((commodity['sm_sort_name'] == '茄类') & ((commodity['name'].str.contains('番茄')) | (commodity['name'].str.contains('西红柿'))))]


account = pd.read_csv(f'{input_path}/account.csv')
# 判断account中code列的数据类型是否为str，如果不是，则转换为str
if not isinstance(account['code'].iloc[0], str):
    account['code'] = account['code'].astype('Int64').astype(str)
# account按commodity中的code进行第一次筛选
account = account[account['code'].isin(commodity['code'])]
# 将account中busdate列的数据类型转换为日期类型，但不带时分秒
account['busdate'] = pd.to_datetime(account['busdate'], format='%Y-%m-%d')
account.sort_values(by=['busdate', 'code'], inplace=True)
# account按日期范围进行第二次筛选
account = account[(account['busdate'] >= first_day) & (account['busdate'] <= last_day)]
account['busdate'] = account['busdate'].apply(lambda x: x.date())
account['unit_cost'] = account['sum_cost'] / account['amount']
account.dropna(subset=['unit_cost'], inplace=True)
account['unit_cost'] = account['unit_cost'].round(2)
# account按unit_cost列进行第三次筛选。以此账表中的code和busdate，作为后续筛选commodity和running的基准。
account = account[account['unit_cost'] > unit_cost_critical]
print(f"account.isnull().sum():\n{account.isnull().sum().T}", '\n')
print(account.info(), '\n')

account.to_csv(f'{output_path_self_use}/account.csv', index=False)
account.rename(columns={'class': '课别', 'code': '单品编码', 'busdate': '日期', 'unit_cost': '当天进货单价(元)'}, inplace=True)
account.drop(columns=['organ', 'sum_cost', 'amount', 'sum_price', 'sum_disc'], inplace=True)
account.to_excel(f'{output_path}/account.xlsx', index=False)


# account中code形成基准后，再次筛选commodity，才能输出，使得commodity中的code与account中的code一致
commodity = commodity[commodity['code'].isin(account['单品编码'])]
print(f"commodity.isnull().sum():\n{commodity.isnull().sum()}", '\n')
print('commodity.info()','\n',commodity.info(), '\n')

commodity.to_csv(f'{output_path_self_use}/commodity.csv', index=False)
commodity.rename(columns={'class': '课别', 'code': '单品编码', 'name': '单品名称', 'sm_sort': '小分类编码', 'md_sort': '中分类编码', 'bg_sort': '大分类编码', 'sm_sort_name': '小分类名称', 'md_sort_name': '中分类名称', 'bg_sort_name': '大分类名称'}, inplace=True)
commodity.to_excel(f'{output_path}/commodity.xlsx', index=False)


running = pd.read_csv(f'{input_path}/running.csv')
if not isinstance(running['code'].iloc[0], str):
    running['code'] = running['code'].astype('Int64').astype(str)
# running按最终形成基准的commodity中的code进行第一次筛选
running = running[running['code'].isin(commodity['单品编码'])]
running['selldate'] = pd.to_datetime(running['selldate'])
running['selldate'] = running['selldate'].apply(lambda x: x.date())
# 将running中selldate和code，与account中日期和单品编码相同的筛选出来
running = running.merge(account[['日期', '单品编码']], how='inner', left_on=['selldate', 'code'], right_on=['日期', '单品编码'])
# merge之后马上剔除多余的列，不能留到后面有相同列名的时候一起剔除，否则会剔除掉account中的日期和单品编码
running.drop(columns=['日期', '单品编码'], inplace=True)
running.sort_values(by=['selldate', 'code'], inplace=True)
running['打折销售'] = ['是' if x > 0 else '否' for x in running['sum_disc']]
assert running['打折销售'].value_counts().values.sum() == running.shape[0], '流水表打折销售列计算有误'

# 如果苹果在commodity的小分类名称中存在，需要输出running表，用于question_3_pre.py
if '苹果' in commodity['小分类名称'].unique():
    running.to_csv(f'{output_path_self_use}/running.csv', index=False)

running.rename(columns={'selldate': '销售日期', 'selltime': '扫码销售时间', 'class': '课别', 'code': '单品编码', 'amount': '销量', 'price': '销售单价(元)', 'type': '销售类型'}, inplace=True)
running.drop(columns=['organ', 'sum_disc', 'sum_sell'], inplace=True)

run_com = pd.merge(running, commodity, on=['课别', '单品编码'], how='left')
print(run_com['小分类名称'].value_counts().sort_values(ascending=False), '\n')
print(f"小分类编码与名称不唯一匹配的个数：{sum(run_com['小分类编码'].value_counts().sort_values(ascending=False).values != run_com['小分类名称'].value_counts().sort_values(ascending=False).values)}", '\n')
print(f"running.isnull().sum():\n{running.isnull().sum()}",'\n')
print('running.info()','\n',running.info(),'\n')

# running.to_csv(f'{output_path}/running.csv', index=False, encoding='utf-8-sig')  # encoding='utf-8-sig'，解决excel打开，中文是乱码的问题
running.to_excel(f'{output_path}/running.xlsx', index=False)
print(running['销售类型'].value_counts().sort_values(ascending=False), '\n')
print(running['打折销售'].value_counts().sort_values(ascending=False), '\n')

print("data_output.py运行完毕！")


# # 统计空值和0值的样本数占比
# print(f"实际订货量为0的样本数占比：{account_order[account_order['order_real'] == 0].shape[0] / account_order.shape[0]}")
# print(f"实际订货量为空值的样本数占比：{account_order[account_order['order_real'].isnull()].shape[0] / account_order.shape[0]}")
# print(f"实际订货量为0以及空值的样本数占比：{account_order[(account_order['order_real'] == 0) | (account_order['order_real'].isnull())].shape[0] / account_order.shape[0]}", '\n')

# # 查看commodity，running，account_order的缺失值情况
# df1 = pd.DataFrame(commodity.isnull().sum(), columns=['null_count'])
# df2 = pd.DataFrame(running.isnull().sum(), columns=['null_count'])
# df3 = pd.DataFrame(account_order.isnull().sum(), columns=['null_count'])
# null_counts = pd.concat([df1, df2, df3], axis=0)
# print(null_counts.T)


# # 查看commodity，running，account_order的数据类型
# df1 = pd.DataFrame(commodity.dtypes, columns=['data_type']).T
# df2 = pd.DataFrame(running.dtypes, columns=['data_type']).T
# df3 = pd.DataFrame(account_order.dtypes, columns=['data_type']).T
# df = pd.concat([df1, df2, df3], axis=0)
# print(df)


# # 查看commodity，running，account_order中，不同字段的唯一值个数
# commodity_info = pd.DataFrame({
#     'code_unique': [commodity['单品名称'].nunique()],
#     'name_unique': [commodity['name'].nunique()],
#     'sm_sort_unique': [commodity['sm_sort'].nunique()],
#     'sm_sort_name_unique': [commodity['sm_sort_name'].nunique()],
#     'md_sort_unique': [commodity['md_sort'].nunique()],
#     'md_sort_name_unique': [commodity['md_sort_name'].nunique()],
#     'bg_sort_unique': [commodity['bg_sort'].nunique()],
#     'bg_sort_name_unique': [commodity['bg_sort_name'].nunique()],
#     'class_unique': [commodity['课别'].nunique()]
# })
# print(commodity_info)

# running_info = pd.DataFrame({
#     'code_unique': [running['单品名称'].nunique()],
#     'class_unique': [running['课别'].nunique()]
# })
# print(running_info)

# unique_counts = pd.DataFrame({
#     'code_unique': [account_order['单品名称'].nunique()],
#     'name_unique': [account_order['name'].nunique()],
#     'class_unique': [account_order['课别'].nunique()]
# })
# print(unique_counts)


# # 查看commodity，running，account_order的统计信息
# print(commodity.describe().T)
# print(running.describe().T)
# print(account_order.describe().T)


# # # 查看running，account_order中，数值型字段的分布情况，并画图显示
# # # running
# # running_num = running.select_dtypes(include=['int64', 'float64'])
# # for i in running_num.columns:
# #     plt.figure(figsize=(10, 5))
# #     sns.set_style("whitegrid")
# #     sns.boxplot(x=running_num[i], color='blue')
# #     sns.despine(left=True)
# #     plt.show()
# #     plt.figure(figsize=(10, 5))
# #     sns.set_style("white")
# #     sns.kdeplot(running_num[i], shade=True, color='blue')
# #     sns.despine(left=True)
# #     plt.show()

# # account_order
# account_order_num = account_order.select_dtypes(include=['int64', 'float64'])
# for i in account_order_num.columns:
#     plt.figure(figsize=(10, 5))
#     sns.set_style("whitegrid")
#     sns.boxplot(x=account_order_num[i], color='blue')
#     sns.despine(left=True)
#     plt.show()
#     plt.figure(figsize=(10, 5))
#     sns.set_style("white")
#     sns.kdeplot(account_order_num[i], shade=True, color='blue')
#     sns.despine(left=True)
#     plt.show()
