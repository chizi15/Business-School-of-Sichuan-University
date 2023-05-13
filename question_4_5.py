# -*- coding: utf-8 -*-
from prophet import Prophet
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 两层dirname才能得到上上级目录
# 添加其他文件夹路径的脚本到系统临时路径，不会保留在环境变量中，每次重新append即可
sys.path.append(base_path)  # regression_evaluation_main所在文件夹的绝对路径
from regression_evaluation_main import regression_evaluation_def as ref
pd.set_option('display.max_rows', 8)


coef = 0.5 # 相关系数排序分组时的阈值
corr_neg = -0.3 # 销量与售价的负相关性阈值
periods = 7 # 预测步数
interval_width = 0.95 # prophet的置信区间宽度

# 读取数据
df = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\account.csv")
df.sort_values(by=['busdate'], inplace=True)
df_students = df[df['busdate'].isin(df['busdate'].unique()[:-periods])]
df_students.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\students_use_data\df_students.xlsx")
df_students.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\students_use_data\df_students.xlsx")
sort = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\commodity.csv")
sort.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\students_use_data\sort.xlsx")
sort.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\students_use_data\sort.xlsx")
# 拼接账表和商品资料表
df = pd.merge(df, sort, how='left', on=['code', 'class'])
df['busdate'] = pd.to_datetime(df['busdate'])
df.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\teachers_use\data\df.xlsx")
df.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\teachers_use\data\df.xlsx")
# 向小分类层级聚合数据
# df_p1 = df[(df['bg_sort_name']=='蔬菜')|(df['sm_sort_name']=='苹果')]
df_p1 = df
df_p1 = df_p1.groupby(['sm_sort','busdate']).agg({'amount': 'mean','sum_price': 'mean', 'sum_cost': 'mean'}).reset_index()
# 计算平均售价、进价和毛利率
df_p1['price'] = df_p1['sum_price'] / df_p1['amount']
df_p1['cost_price'] = df_p1['sum_cost'] / df_p1['amount']
df_p1['profit'] = (df_p1['price'] - df_p1['cost_price']) / df_p1['price']
sale_sm = df_p1.dropna()
sale_sm = sale_sm[sale_sm['profit'] >= 0]
sale_sm.sort_values(by=['sm_sort', 'busdate'], inplace=True)
print(f'总共有{sale_sm["sm_sort"].nunique()}个小分类')


# question_4
# 在df_p1中，对各个sm_sort分别画时间序列图，横坐标是busdate，纵坐标是amount
for code, data in sale_sm.groupby(['sm_sort']):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(data['busdate'], data['amount'])
    plt.title(f'{code}')
    plt.show()
    fig.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\results\sm_sort\%s.svg" % code)
fig.clear()

# 筛选销量与价格负相关性强的小分类
typeA = []
typeB = []
for code, data in sale_sm.groupby(['sm_sort']):
    if len(data)>5:
        r = ss.spearmanr(data['amount'], data['price']).correlation
        if r < corr_neg:
            typeA.append(code)
        else:
            typeB.append(code)
sale_sm_a = sale_sm[sale_sm['sm_sort'].isin(typeA)]
sale_sm_b = sale_sm[sale_sm['sm_sort'].isin(typeB)]
print(f'销量与价格的负相关性强的小分类一共有{sale_sm_a["sm_sort"].nunique()}个')
print(f'销量与价格的负相关性弱的小分类一共有{sale_sm_b["sm_sort"].nunique()}个')
sale_sm_a.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\results\销量与价格的负相关性强的小分类的销售数据.xlsx")
sale_sm_b.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\results\销量与价格的负相关性弱的小分类的销售数据.xlsx")

# 行转列
sale_sm_a_t = pd.pivot(sale_sm_a, index="busdate", columns="sm_sort", values="amount")
# 正态化，使样本更符合pearson相关性检验的假设
sale_sm_a_t = sale_sm_a_t.apply(lambda x: np.log1p(x), axis=0)
# 计算每列间的相关性
sale_sm_a_coe = sale_sm_a_t.corr(method='pearson') # Compute pairwise correlation of columns, excluding NA/null values
# 画相关系数矩阵的热力图，并保存输出，每个小分类的名字都显示出来，排列稀疏
plt.figure(figsize=(20, 20))
sns.heatmap(sale_sm_a_coe, annot=True, xticklabels=True, yticklabels=True)
plt.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\results\销量与价格的负相关性强的一组中，各个小分类销量的corr_heatmap.svg")

# 对typeA中小分类按相关系数的排序进行分组
# 选择大于相关性大于coef的组合
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
print(f'进行相关性排序并分组后的结果\n{groups_}')

# 将groups_中的集合转换为列表
groups_ = [list(group) for group in groups_]
groups_.append(typeB)
print(f'最终分组结果\n{groups_}')
# 将groups_中的列表转换为df，索引为组号，列名为各个小分类名
groups_df = pd.DataFrame(pd.Series(groups_), columns=['sm_sort'])
groups_df['group'] = groups_df.index+1
# 改变列的顺序
groups_df = groups_df[['group', 'sm_sort']]
groups_df.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\results\相关性分组结果.xlsx", index=False, sheet_name='最后一组是销量对价格不敏感的，前面几组是销量对价格敏感的')

# 对groups_中的每个组，从df_p1中筛选出对应的数据，组成list_df
list_df = [df_p1[df_p1['sm_sort'].isin(group)] for group in groups_]
# 循环对list_df中每个df按busdate进行合并groupby，并求均值
list_df_avg = [df.groupby(['busdate']).agg({'amount': 'mean', 'sum_price': 'mean', 'sum_cost': 'mean'}).reset_index() for df in list_df]
# 对list_df_avg中每个df画时间序列图，横坐标是busdate，纵坐标是amount，图名从组1到组7依次命名
for i, df in enumerate(list_df_avg):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(df['busdate'], df['amount'])
    plt.title(f'group{i+1}')
    plt.show()
    fig.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_4\results\groups\group%s.svg" % (i+1))
fig.clear()


# question5
seasonality_mode = 'additive'
holidays_prior_scale=10
seasonality_prior_scale=10
holiday = True
weekly = True
yearly = True
monthly = True
quarterly = True
weekly_fourier_order = 10
yearly_fourier_order = 3
monthly_fourier_order = 2
quarterly_fourier_order = 2
weekly_prior_scale = 10
yearly_prior_scale = 10
monthly_prior_scale = 1
quarterly_prior_scale = 1
mcmc_samples = 100
type_ = ['amount', 'price', 'profit']

# 将sale_sm按sm_sort进行分组切分，将得到的每组df赋给一个list
list_df = [df for _, df in sale_sm.groupby('sm_sort')]
# 将list_df中每个df的sm_sort取出来，组成一个列表
sm_sort = [list_df[i]['sm_sort'].unique()[0] for i in range(len(list_df))]


def prophet_model(df, periods, seasonality_mode, holidays_prior_scale, seasonality_prior_scale, holiday, weekly, yearly, monthly, quarterly, weekly_fourier_order, yearly_fourier_order, monthly_fourier_order, quarterly_fourier_order, weekly_prior_scale, yearly_prior_scale, monthly_prior_scale, quarterly_prior_scale, mcmc_samples, type_, sm_sort):
    m = Prophet(seasonality_mode=seasonality_mode, holidays_prior_scale=holidays_prior_scale, seasonality_prior_scale=seasonality_prior_scale, mcmc_samples=mcmc_samples, interval_width=interval_width)
    if holiday:
        m.add_country_holidays(country_name='CN')
    if weekly:
        m.add_seasonality(name='weekly', period=7, fourier_order=weekly_fourier_order, prior_scale=weekly_prior_scale)
    if yearly:
        m.add_seasonality(name='yearly', period=365, fourier_order=yearly_fourier_order, prior_scale=yearly_prior_scale)
    if monthly:
        m.add_seasonality(name='monthly', period=30.5, fourier_order=monthly_fourier_order, prior_scale=monthly_prior_scale)
    if quarterly:
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=quarterly_fourier_order, prior_scale=quarterly_prior_scale)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    fig1 = m.plot(forecast, uncertainty=True)
    # plt.show()
    fig2 = m.plot_components(forecast, uncertainty=True)
    # plt.show()
    # 如果在"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\"中不存在sm_sort编号的文件夹，则创建
    if not os.path.exists(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\%s" % sm_sort):
        os.mkdir(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\%s" % sm_sort)
    if not os.path.exists(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\{}\{}".format(sm_sort, type_)): # 两级目录要分两次创建，n级目录要分n次创建
        os.mkdir(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\{}\{}".format(sm_sort, type_))
    fig1.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\{}\{}\fit.svg".format(sm_sort, type_), dpi=300, bbox_inches='tight')
    fig2.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\{}\{}\fit_components.svg".format(sm_sort, type_), dpi=300, bbox_inches='tight')
    fig1.clear()
    fig2.clear()

    return forecast


# 用prophet模型对每组df进行预测
list_forecast = []
for tp in type_:
    list_forecast.append([prophet_model(df[:-periods][['busdate', tp]].rename(columns={'busdate': 'ds', tp: 'y'}), periods, seasonality_mode, holidays_prior_scale, seasonality_prior_scale, holiday, weekly, yearly, monthly, quarterly, weekly_fourier_order, yearly_fourier_order, monthly_fourier_order, quarterly_fourier_order, weekly_prior_scale, yearly_prior_scale, monthly_prior_scale, quarterly_prior_scale, mcmc_samples, type_=tp, sm_sort=df['sm_sort'].unique()[0]) for df in list_df])

# 输出预测结果
for i in range(len(sm_sort)):
    for j in range(len(type_)-1):
        list_forecast[j][i][-periods:].to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_5\results\{}\{}\forecast.xlsx".format(sm_sort[i], type_[j]), index=False)


# df1 = list_forecast[j][i-1][['ds', 'yhat']][-periods:]
# df2 = list_df[i-1][['sm_sort', 'busdate', 'amount', 'price', 'profit']][-periods:]

# price = list_forecast[1][0]['yhat'][-periods:] / list_forecast[0][0]['yhat'][-periods:]
# cost_price = list_forecast[2][0]['yhat'][-periods:]
# profit = (price - cost_price) / price

# price = []
# cost_price = []
# profit = []

# for i in range(len(sm_sort)):
#     price.append(list_forecast[1][i]['yhat'][-periods:] / list_forecast[0][i]['yhat'][-periods:])
#     cost_price.append(list_forecast[2][i]['yhat'][-periods:])
#     profit.append((price[i] - cost_price[i]) / price[i] if )
