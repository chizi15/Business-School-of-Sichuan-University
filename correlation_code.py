import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from data_output import output_path_self_use


input_path = output_path_self_use
script_name = os.path.basename(__file__).split('.')[0]
base_path = "D:\\Work info\\SCU\\MathModeling\\2023\\data\\processed\\" + script_name + "\\"
output_path = os.path.join(base_path)
os.makedirs(output_path, exist_ok=True)


code_sm = pd.read_excel(f"{input_path}" + "附件1-单品-分类.xlsx", sheet_name="Sheet1")
code_sm[['单品编码', '分类编码']] = code_sm[['单品编码', '分类编码']].astype(str)
code_sm.rename(columns={'单品编码': 'code', '单品名称': 'name', '分类编码': 'sm_sort', '分类名称': 'sm_sort_name'}, inplace=True)
print(code_sm.dtypes, '\n')

run = pd.read_excel(f"{input_path}" + "附件2-流水-销量-售价.xlsx", sheet_name="Sheet1")
run['单品编码'] = run['单品编码'].astype(str)
run_seg = run[run['销售类型'] == '销售'].copy()
run_seg.drop(columns=['销售单价(元/千克)', '销售类型', '是否打折销售', '扫码销售时间'], inplace=True)
run_seg.rename(columns={'销售日期': 'busdate', '单品编码': 'code', '销量(千克)': 'amount'}, inplace=True)
print(run_seg.dtypes, '\n')

# 根据流水表run_seg按日聚合和商品资料表code_sm拼接得到账表account
account = run_seg.groupby(['busdate', 'code'])['amount'].sum().reset_index()
account = account.merge(code_sm, how='left', on='code')

code_count = account.groupby('code').count().reset_index()
code_count = code_count[(code_count['busdate'] <= 28) & (code_count['busdate'] > 14)]
code_count = code_count[['code']]
account = account.merge(code_count, how='right', on='code')

# 对筛选出的account中，各个不同的code用sns绘制销量随时间的变化图，并计算各个code两两对应的spearmann相关系数，并画出热力图
# 对数据按照 'code' 进行分组
grouped = account.groupby('name')
# tuple(grouped)
# 对每个组绘制销量随时间的变化图
for name, group in grouped:
    group['busdate'] = pd.to_datetime(group['busdate'])
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=group['busdate'], y=group['amount'])
    sns.scatterplot(x=group['busdate'], y=group['amount'], color='r', label='销售日')
    plt.title(f"{name}_销量日序")
    plt.xlabel('销售日期')
    plt.ylabel('销量')
    plt.xlim(run_seg['busdate'].min(), run_seg['busdate'].max())
    plt.savefig(f"{output_path}" + f"{name}_销量日序.svg")
    plt.show()

# 计算每两个 'name' 之间的 Spearman 相关系数
names = account['name'].unique()
correlation_matrix = pd.DataFrame(index=names, columns=names)

for name1 in names:
    for name2 in names:
        try:
            correlation, _ = spearmanr(account[account['name'] == name1]['amount'], account[account['name'] == name2]['amount'])
            correlation_matrix.loc[name1, name2] = correlation
        except:
            correlation_matrix.loc[name1, name2] = np.nan

correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors='coerce')
# 绘制热力图
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("稀疏单品间的Spearman相关系数热力图", fontsize=20, loc='center')
plt.xticks(rotation=45, fontsize=15)
plt.yticks(rotation=45, fontsize=15)
plt.savefig(f"{output_path}" + "稀疏单品间的Spearman相关系数热力图.svg")
plt.show()


# 获取年份、季度、月份和周数
account['year'] = account['busdate'].dt.year
account['month'] = account['busdate'].dt.month
account['week'] = account['busdate'].dt.isocalendar().week
account['quarter'] = account['busdate'].dt.quarter
# 将年份、月份和周数组合成一个新的列
account['year_week'] = account['year'].astype(str) + '-' + account['week'].astype(str)
account['year_month'] = account['year'].astype(str) + '-' + account['month'].astype(str)
account['year_quarter'] = account['year'].astype(str) + '-' + account['quarter'].astype(str)
# 删除临时列
account.drop(['year', 'quarter', 'month', 'week'], axis=1, inplace=True)

name_week = account.groupby(['name', 'year_week']).agg({'amount': 'sum'}).reset_index()
grouped = name_week.groupby(['name'])
for name, group in grouped:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=group['year_week'], y=group['amount'])
    sns.scatterplot(x=group['year_week'], y=group['amount'], color='r', label='销售周')
    plt.legend(loc='best')
    plt.title(f"{name}_销量周序")
    plt.xlabel('销售年-周')
    plt.ylabel('销量')
    plt.xlim(group['year_week'].min(), group['year_week'].max())
    plt.show()


# def ts_corr(df, date_col='year_week', amount_col='amount'):
#     account = df.copy()
#     # 对筛选出的account中，各个不同的code用sns绘制销量随时间的变化图，并计算各个code两两对应的spearmann相关系数，并画出热力图
#     # 对数据按照 'code' 进行分组
#     grouped = account.groupby('name')
#     # tuple(grouped)
#     # 对每个组绘制销量随时间的变化图
#     for name, group in grouped:
#         # group[date_col] = pd.to_datetime(group[date_col])
#         plt.figure(figsize=(10, 6))
#         sns.lineplot(x=group[date_col], y=group['amount'])
#         sns.scatterplot(x=group[date_col], y=group['amount'], color='r', label='销售日')
#         plt.title(f"{name}_销量日序")
#         plt.xlabel('销售日期')
#         plt.ylabel('销量')
#         plt.xlim(run_seg[date_col].min(), run_seg[date_col].max())
#         plt.savefig(f"{output_path}" + f"{name}_销量日序.svg")
#         plt.show()

#     # 计算每两个 'name' 之间的 Spearman 相关系数
#     names = account['name'].unique()
#     correlation_matrix = pd.DataFrame(index=names, columns=names)

#     for name1 in names:
#         for name2 in names:
#             try:
#                 correlation, _ = spearmanr(account[account['name'] == name1]['amount'], account[account['name'] == name2]['amount'])
#                 correlation_matrix.loc[name1, name2] = correlation
#             except:
#                 correlation_matrix.loc[name1, name2] = np.nan

#     correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors='coerce')
#     # 绘制热力图
#     plt.figure(figsize=(20, 20))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#     plt.title("稀疏单品间的Spearman相关系数热力图", fontsize=20, loc='center')
#     plt.xticks(rotation=45, fontsize=15)
#     plt.yticks(rotation=45, fontsize=15)
#     plt.savefig(f"{output_path}" + "稀疏单品间的Spearman相关系数热力图.svg")
#     plt.show()


# ts_corr(account, date_col='year_week', amount_col='amount')