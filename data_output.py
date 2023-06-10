import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 6)


input_path = r'D:\Work info\WestUnion\data\processed\HLJ\脱敏及筛选后样本数据\output'
output_path = r"D:\Work info\SCU\MathModeling\2023\data\output"
output_path_self_use = r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata"


commodity = pd.read_csv(f'{input_path}/commodity.csv')
commodity.to_csv(f'{output_path_self_use}/commodity.csv', index=False)
commodity.to_excel(f'{output_path}/commodity.xlsx', index=False)

order = pd.read_csv(f'{input_path}/订货数据.csv')
order.drop(columns=['order_pred', 'loss_theory'], inplace=True)
order['busdate'] = pd.to_datetime(order['busdate'])
order.sort_values(by=['busdate', 'code'], inplace=True)

account = pd.read_csv(f'{input_path}/account.csv')
account['busdate'] = pd.to_datetime(account['busdate'])
account.sort_values(by=['busdate', 'code'], inplace=True)
account = account[account['busdate'] >= order['busdate'].min()]
account.to_csv(f'{output_path_self_use}/account.csv', index=False)
account['unit_cost'] = account['sum_cost'] / account['amount']
account.drop(columns=['amount', 'sum_price', 'sum_disc', 'sum_cost'], inplace=True)

running = pd.read_csv(f'{input_path}/running.csv')
running['selldate'] = pd.to_datetime(running['selldate'])
running.sort_values(by=['selldate', 'code'], inplace=True)
running = running[running['selldate'] >= order['busdate'].min()]
running.to_csv(f'{output_path_self_use}/running.csv', index=False)
running.drop(columns=['sum_disc', 'sum_sell'], inplace=True)
# running.to_csv(f'{output_path}/running.csv', index=False, encoding='utf-8-sig')  # encoding='utf-8-sig'，解决excel打开，中文是乱码的问题
running.to_excel(f'{output_path}/running.xlsx', index=False)

# 将account和order按['organ', 'class', 'code', 'busdate']合并
account_order = pd.merge(account, order, on=['organ', 'class', 'code', 'busdate'], how='left')
account_order.to_excel(f'{output_path}/account_order.xlsx', index=False)


# 统计空值和0值的样本数占比
print(f"实际订货量为0的样本数占比：{account_order[account_order['order_real'] == 0].shape[0] / account_order.shape[0]}")
print(f"实际订货量为空值的样本数占比：{account_order[account_order['order_real'].isnull()].shape[0] / account_order.shape[0]}")
print(f"实际订货量为0以及空值的样本数占比：{account_order[(account_order['order_real'] == 0) | (account_order['order_real'].isnull())].shape[0] / account_order.shape[0]}", '\n')

# 查看commodity，running，account_order的缺失值情况
df1 = pd.DataFrame(commodity.isnull().sum(), columns=['null_count'])
df2 = pd.DataFrame(running.isnull().sum(), columns=['null_count'])
df3 = pd.DataFrame(account_order.isnull().sum(), columns=['null_count'])
null_counts = pd.concat([df1, df2, df3], axis=0)
print(null_counts.T)


# 查看commodity，running，account_order的数据类型
df1 = pd.DataFrame(commodity.dtypes, columns=['data_type']).T
df2 = pd.DataFrame(running.dtypes, columns=['data_type']).T
df3 = pd.DataFrame(account_order.dtypes, columns=['data_type']).T
df = pd.concat([df1, df2, df3], axis=0)
print(df)


# 查看commodity，running，account_order中，不同字段的唯一值个数
commodity_info = pd.DataFrame({
    'code_unique': [commodity['code'].nunique()],
    'name_unique': [commodity['name'].nunique()],
    'sm_sort_unique': [commodity['sm_sort'].nunique()],
    'sm_sort_name_unique': [commodity['sm_sort_name'].nunique()],
    'md_sort_unique': [commodity['md_sort'].nunique()],
    'md_sort_name_unique': [commodity['md_sort_name'].nunique()],
    'bg_sort_unique': [commodity['bg_sort'].nunique()],
    'bg_sort_name_unique': [commodity['bg_sort_name'].nunique()],
    'class_unique': [commodity['class'].nunique()]
})
print(commodity_info)

running_info = pd.DataFrame({
    'code_unique': [running['code'].nunique()],
    'class_unique': [running['class'].nunique()]
})
print(running_info)

unique_counts = pd.DataFrame({
    'code_unique': [account_order['code'].nunique()],
    'name_unique': [account_order['name'].nunique()],
    'class_unique': [account_order['class'].nunique()]
})
print(unique_counts)


# 查看commodity，running，account_order的统计信息
print(commodity.describe().T)
print(running.describe().T)
print(account_order.describe().T)


# # 查看running，account_order中，数值型字段的分布情况，并画图显示
# # running
# running_num = running.select_dtypes(include=['int64', 'float64'])
# for i in running_num.columns:
#     plt.figure(figsize=(10, 5))
#     sns.set_style("whitegrid")
#     sns.boxplot(x=running_num[i], color='blue')
#     sns.despine(left=True)
#     plt.show()
#     plt.figure(figsize=(10, 5))
#     sns.set_style("white")
#     sns.kdeplot(running_num[i], shade=True, color='blue')
#     sns.despine(left=True)
#     plt.show()

# account_order
account_order_num = account_order.select_dtypes(include=['int64', 'float64'])
for i in account_order_num.columns:
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    sns.boxplot(x=account_order_num[i], color='blue')
    sns.despine(left=True)
    plt.show()
    plt.figure(figsize=(10, 5))
    sns.set_style("white")
    sns.kdeplot(account_order_num[i], shade=True, color='blue')
    sns.despine(left=True)
    plt.show()
