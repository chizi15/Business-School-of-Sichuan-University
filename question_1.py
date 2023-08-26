# -*- coding: utf-8 -*-
from prophet import Prophet
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_output import output_path_self_use, first_day, last_day
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 8)


coef = 0.5 # 相关系数排序分组时的阈值
corr_neg = -0.3 # 销量与售价的负相关性阈值


input_path = output_path_self_use + "\\"
output_path = r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1" + "\\"
if not os.path.exists(output_path):
    os.makedirs(output_path)


# 读取数据
df = pd.read_csv(input_path + "account.csv")
df.sort_values(by=['busdate'], inplace=True)

# 输出这三条时序图中，非空数据的起止日期，用循环实现
for col in ['amount', 'sum_cost', 'sum_price']:
    print(f'{col}非空数据的起止日期为：{df[df[col].notnull()]["busdate"].min()}到{df[df[col].notnull()]["busdate"].max()}', '\n')

# 断言df中数值型字段的起止日期相同
assert (df[df['amount'].notnull()]["busdate"].min() == df[df['sum_cost'].notnull()]["busdate"].min() == df[df['sum_price'].notnull()]["busdate"].min()), "三个字段非空数据的开始日期不相同"
assert (df[df['amount'].notnull()]["busdate"].max() == df[df['sum_cost'].notnull()]["busdate"].max() == df[df['sum_price'].notnull()]["busdate"].max()), "三个字段非空数据的结束日期不相同"


sort = pd.read_csv(input_path + "commodity.csv")
# 拼接账表和商品资料表
df = pd.merge(df, sort, how='left', on=['code', 'class'])
df['busdate'] = pd.to_datetime(df['busdate'])
df.drop(columns=['sum_disc'], inplace=True)


# 向小分类层级聚合数据
# df_p1 = df[(df['bg_sort_name']=='蔬菜')|(df['sm_sort_name']=='苹果')]
df_p1 = df
print(df_p1.dtypes, '\n')
print(sort.dtypes, '\n')
df_p1 = df_p1.groupby(['sm_sort', 'busdate']).agg({'amount': 'mean','sum_price': 'mean', 'sum_cost': 'mean'}).reset_index()
df_p1 = df_p1.merge(sort.drop(columns=['class', 'code', 'name']).drop_duplicates(), on='sm_sort', how='left')

# 计算平均售价、进价和毛利率
df_p1['price'] = df_p1['sum_price'] / df_p1['amount']
df_p1['cost_price'] = df_p1['sum_cost'] / df_p1['amount']
df_p1['profit'] = (df_p1['price'] - df_p1['cost_price']) / df_p1['price']
sale_sm = df_p1.dropna()
sale_sm = sale_sm[sale_sm['profit'] >= 0]
sale_sm.sort_values(by=['sm_sort', 'busdate'], inplace=True)
print(f'总共有{sale_sm["sm_sort"].nunique()}个小分类', '\n')
# 判断sale_sm['sm_sort']中是否有小分类的名称中包含'.'，或者sale_sm['sm_sort']的数据类型是否为float64
if sale_sm['sm_sort'].dtype == 'float64' or sale_sm['sm_sort'].astype(str).str.contains('\.').any():
    print("sale_sm['sm_sort'] is of type float64 or contains decimal points.")
    sale_sm['sm_sort'] = sale_sm['sm_sort'].astype(str).str.split('.').str[0]
else:
    print("sale_sm['sm_sort'] is not of type float64 and does not contain decimal points.")


# 在df_p1中，对各个sm_sort分别画时间序列图，横坐标是busdate，纵坐标是amount
for name, data in sale_sm.groupby(['sm_sort_name']):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(data['busdate'], data['amount'])
    plt.title(f'{name}')
    plt.show()
    fig.savefig(output_path + "%s_销量时序.svg" % name)
    fig.clear()

# 正态化，使样本更符合pearson相关性检验的假设
sale_sm['amount'] = sale_sm['amount'].apply(lambda x: np.log1p(x))
# 筛选销量与价格负相关性强的小分类
typeA = []
typeB = []
for code, data in sale_sm.groupby(['sm_sort_name']):
    if len(data)>5:
        r = ss.spearmanr(data['amount'], data['price']).correlation
        if r < corr_neg:
            typeA.append(code)
        else:
            typeB.append(code)
# 对sale_sm['amount']做np.log1p的逆变换，使数据回到原来的尺度
sale_sm['amount'] = sale_sm['amount'].apply(lambda x: np.expm1(x))
sale_sm_a = sale_sm[sale_sm['sm_sort_name'].isin(typeA)]
sale_sm_b = sale_sm[sale_sm['sm_sort_name'].isin(typeB)]
print(f'销量与价格的负相关性强(小于{corr_neg})的小分类一共有{sale_sm_a["sm_sort_name"].nunique()}个')
print(f'销量与价格的负相关性弱(大于等于{corr_neg})的小分类一共有{sale_sm_b["sm_sort_name"].nunique()}个', '\n')
sale_sm_a.to_excel(output_path + f"小分类销售数据_销量与价格的负相关性强(小于{corr_neg})的一组.xlsx")
sale_sm_b.to_excel(output_path + f"小分类销售数据_销量与价格的负相关性弱(大于等于{corr_neg})的一组.xlsx")

# 计算负相关性强的小分类序列的相关系数并画热力图。
# 先对df行转列
sale_sm_a_t = pd.pivot(sale_sm_a, index="busdate", columns="sm_sort_name", values="amount")
# 计算每列间的相关性
sale_sm_a_coe = sale_sm_a_t.corr(method='pearson') # Compute pairwise correlation of columns, excluding NA/null values
# 画相关系数矩阵的热力图，并保存输出，每个小分类的名字都显示出来，排列稀疏
plt.figure(figsize=(20, 20))
sns.heatmap(sale_sm_a_coe, annot=True, xticklabels=True, yticklabels=True)
plt.savefig(output_path + "小分类销量与价格负相关性强的一组中，各个小分类销量间的corr_heatmap.svg")

# 对typeA中小分类按相关系数的排序进行分组
# 选择相关性大于coef的组合
groups = []
idxs = sale_sm_a_coe.index.to_list()
for idx, row in sale_sm_a_coe.iterrows():
    group = row[row > coef].index.to_list()
    groups.append(group)
# 删除重复使用的小分类
groups_ = []
for group in groups:
    diff_group = []
    for idx in group:
        if idx in idxs:
            idxs.remove(idx)
        else:
            diff_group.append(idx)
    group = set(group)-set(diff_group)
    if group:
        groups_.append(group)
print(f'进行相关性排序，并以相关系数大于{coef}为条件进行分组后的结果\n{groups_}')

# 将groups_中的集合转换为列表
groups_ = [list(group) for group in groups_]
groups_.append(typeB)
print(f'最终分组结果\n{groups_}')
# 将groups_中的列表转换为df，索引为组号，列名为各个小分类名
groups_df = pd.DataFrame(pd.Series(groups_), columns=['sm_sort_name'])
groups_df['group'] = groups_df.index+1
# 改变列的顺序
groups_df = groups_df[['group', 'sm_sort_name']]
groups_df.to_excel(output_path + f"小分类_相关性分组结果：以相关系数大于{coef}为条件.xlsx", index=False, sheet_name='最后一组是销量对价格不敏感的，前面若干组是销量对价格敏感的')

# 对groups_中的每个组，从df_p1中筛选出对应的数据，组成list_df
list_df = [df_p1[df_p1['sm_sort_name'].isin(group)] for group in groups_]
# 循环对list_df中每个df按busdate进行合并groupby，并求均值
list_df_avg = [df.groupby(['busdate']).agg({'amount': 'mean', 'sum_price': 'mean', 'sum_cost': 'mean'}).reset_index() for df in list_df]
# 对list_df_avg中每个df画时间序列图，横坐标是busdate，纵坐标是amount，图名从组1到组7依次命名
for i, df in enumerate(list_df_avg):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(df['busdate'], df['amount'])
    plt.title(f'{groups_[i]}')
    plt.show()
    fig.savefig(output_path + f"{groups_[i]}_按相关性分组合并后的小分类销量时序.svg")
    fig.clear()
