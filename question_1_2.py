# -*- coding: utf-8 -*-
from prophet import Prophet
import pandas as pd
import numpy as np
from scipy import stats
import chinese_calendar
import fitter
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
base_path = os.path.dirname(os.path.dirname(__file__))  # 两层dirname才能得到上上级目录
# 添加其他文件夹路径的脚本到系统临时路径，不会保留在环境变量中，每次重新append即可
# sys.path.append("D:/Work info/Repositories")
sys.path.append(base_path)  # regression_evaluation_main所在文件夹的绝对路径
from regression_evaluation_main import regression_evaluation_def as ref
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 6)
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
"""
中文字体	    说明
‘SimHei’	中文黑体
‘Kaiti’	    中文楷体
‘LiSu’	    中文隶书
‘FangSong’	中文仿宋
‘YouYuan’	中文幼圆
’STSong‘    华文宋体
"""
print('Imported packages successfully.', '\n')


# 设置全局参数
periods = 7 # 预测步数
output_index = 1  # 将前output_index个预测结果作为最终结果
extend_power = 1/5 # 数据扩增的幂次
interval_width = 0.95 # prophet的置信区间宽度
mcmc_samples = 100 # mcmc采样次数


# read and summerize data
account = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\account.csv")
account['busdate'] = pd.to_datetime(account['busdate'])
account.sort_values(by=['busdate'], inplace=True)
account['code'] = account['code'].astype('str')
acct_grup = account.groupby(["organ", "code"])
print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n{account.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_grup)}\n')
print(f"account['sum_disc'].mean(): {account['sum_disc'].mean()}")
print(f"account['sum_price'].mean(): {account['sum_price'].mean()}", '\n')
account.drop(columns=['sum_disc'], inplace=True)


commodity = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\commodity.csv")
commodity[['class', 'bg_sort', 'md_sort', 'sm_sort', 'code']] = commodity[['class', 'bg_sort', 'md_sort', 'sm_sort', 'code']].astype('str')
comodt_grup = commodity.groupby(['code'])
print(f'\ncommodity\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n{commodity.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(commodity.isnull().T.any())}\n\nnumber of commodities:\n{len(comodt_grup)}\n')


# 将account和commodity按['class', 'code']合并
account_commodity = pd.merge(account, commodity, on=['class', 'code'], how='left')
print(f'\naccount_commodity\n\nshape: {account_commodity.shape}\n\ndtypes:\n{account_commodity.dtypes}\n\nisnull-columns:\n{account_commodity.isnull().any()}', '\n')
# account_commodity.to_csv(r"D:\Work info\SCU\MathModeling\2023\data\processed\account_commodity.csv", index=False)


# 将account_commodity按['organ', 'class', 'bg_sort', 'md_sort']进行聚合，得到一张对account_commodity中float类型字段取均值的新表，并且是单行索引
account_commodity_mean = account_commodity.groupby(['organ', 'class', 'bg_sort', 'bg_sort_name', 'md_sort', 'md_sort_name', 'sm_sort', 'sm_sort_name', 'busdate'], as_index=False).mean()
print(f'\naccount_commodity_mean\n\nshape: {account_commodity_mean.shape}\n\ndtypes:\n{account_commodity_mean.dtypes}\n\nisnull-columns:\n{account_commodity_mean.isnull().any()}', '\n')


# question_1
sm_qielei_all = account_commodity_mean[account_commodity_mean['sm_sort_name'] == "茄类"]
sm_qielei_all = sm_qielei_all.round(3)
sm_qielei_all.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\teachers_use\data\sm_qielei_all.xlsx", index=False, sheet_name='茄类在全集上的样本')
# 获取茄类在训练集上的样本
sm_qielei = sm_qielei_all[:-periods]
# 将训练集上负毛利的异常样本剔除
sm_qielei = sm_qielei[sm_qielei['sum_cost'] <= sm_qielei['sum_price']]
sm_qielei = sm_qielei.round(3)
sm_qielei.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\students_use_data\sm_qielei_train.xlsx", index=False, sheet_name='茄类在训练集上的样本')

# 对sm_qielei_all中数值型变量的列：amount，sum_cost和sum_price画时序图
sm_qielei_all_num = sm_qielei_all.select_dtypes(include=np.number)
for col in sm_qielei_all_num:
    sns.lineplot(x='busdate', y=col, data=sm_qielei_all)
    plt.xticks(rotation=45)
    plt.xlabel('busdate')
    plt.ylabel(col)
    plt.title('Time Series Graph')
    plt.show()

# 输出这三条时序图中，非空数据的起止日期，用循环实现
for col in sm_qielei_all_num:
    print(f'{col}非空数据的起止日期为：{sm_qielei_all[sm_qielei_all[col].notnull()]["busdate"].min()}到{sm_qielei_all[sm_qielei_all[col].notnull()]["busdate"].max()}', '\n')

# 断言这三个字段非空数据的起止日期相同
assert (sm_qielei_all[sm_qielei_all['amount'].notnull()]["busdate"].min() == sm_qielei_all[sm_qielei_all['sum_cost'].notnull()]["busdate"].min() == sm_qielei_all[sm_qielei_all['sum_price'].notnull()]["busdate"].min()), "三个字段非空数据的开始日期不相同"
assert (sm_qielei_all[sm_qielei_all['amount'].notnull()]["busdate"].max() == sm_qielei_all[sm_qielei_all['sum_cost'].notnull()]["busdate"].max() == sm_qielei_all[sm_qielei_all['sum_price'].notnull()]["busdate"].max()), "三个字段非空数据的结束日期不相同"

# 以'busdate'为横坐标，'amount'为纵坐标，画出时序图。用sns来画图，使图更美观，使横坐标的日期不会重叠，并且横坐标以每月为时间间隔显示
sns.lineplot(x='busdate', y='amount', data=sm_qielei)
plt.xticks(rotation=45)
plt.xlabel('busdate')
plt.ylabel('amount')
plt.title('Time Series Graph')
plt.show()

# 用prophet获取训练集上的星期效应系数、节日效应系数和年季节性效应系数
qielei_prophet_amount = sm_qielei[['busdate', 'amount']].rename(columns={'busdate': 'ds', 'amount': 'y'})
m_amount = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=mcmc_samples, interval_width=interval_width)
m_amount.add_country_holidays(country_name='CN')
m_amount.fit(qielei_prophet_amount)
future_amount = m_amount.make_future_dataframe(periods=periods)
forecast_amount = m_amount.predict(future_amount)
fig1 = m_amount.plot(forecast_amount)
plt.show()
fig2 = m_amount.plot_components(forecast_amount)
plt.show()
fig1.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\fit_amount.svg", dpi=300, bbox_inches='tight')
fig2.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\fit_amount_components.svg", dpi=300, bbox_inches='tight')

holiday_effect = forecast_amount[['ds', 'holidays', 'holidays_lower', 'holidays_upper']]
weekly_effect = forecast_amount[['ds', 'weekly', 'weekly_lower', 'weekly_upper']]
yearly_effect = forecast_amount[['ds', 'yearly', 'yearly_lower', 'yearly_upper']]
multiplicative_terms = forecast_amount[['ds', 'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper']]

# 根据qielei_prophet的ds列，将holiday_effect, weekly_effect, yearly_effect, multiplicative_terms左连接合并到qielei_prophet中
qielei_prophet_amount = pd.merge(qielei_prophet_amount, holiday_effect, on='ds', how='left')
qielei_prophet_amount = pd.merge(qielei_prophet_amount, weekly_effect, on='ds', how='left')
qielei_prophet_amount = pd.merge(qielei_prophet_amount, yearly_effect, on='ds', how='left')
qielei_prophet_amount = pd.merge(qielei_prophet_amount, multiplicative_terms, on='ds', how='left')
qielei_prophet_amount = qielei_prophet_amount.rename(columns={'holidays': 'holiday_effect', 'weekly': 'weekly_effect', 'yearly': 'yearly_effect', 'multiplicative_terms': 'total_effect'})
print(qielei_prophet_amount.isnull().sum(), '\n')

# 在qielei_prophet中，增加中国日历的星期和节假日的列，列名分别为weekday和holiday
qielei_prophet_amount['weekday'] = qielei_prophet_amount['ds'].dt.weekday # 0-6, Monday is 0
# 根据日期获取中国节日名称，使用chinese_calendar库
qielei_prophet_amount['holiday'] = qielei_prophet_amount['ds'].apply(lambda x: chinese_calendar.get_holiday_detail(x)[1] if chinese_calendar.get_holiday_detail(x)[0] else None)

# 保存输出带有时间效应和星期、节假日标签的茄类销量样本
qielei_prophet_amount = qielei_prophet_amount.round(3)
qielei_prophet_amount.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\销量时序分解.xlsx", index=False, sheet_name='用历史销量计算出的时间效应，合并到训练集中')

# 验证prophet分解出的各个分项的计算公式
print(f"在乘法模式下，trend*(1+multiplicative_terms)=yhat, 即：sum(forecast['trend']*(1+forecast['multiplicative_terms'])-forecast['yhat']) = {sum(forecast_amount['trend']*(1+forecast_amount['multiplicative_terms'])-forecast_amount['yhat'])}", '\n')
print(f"sum(forecast['multiplicative_terms']-(forecast['holidays']+forecast['weekly']+forecast['yearly'])) = {sum(forecast_amount['multiplicative_terms']-(forecast_amount['holidays']+forecast_amount['weekly']+forecast_amount['yearly']))}", '\n')

# 剔除sm_qielei的时间相关效应，得到sm_qielei_no_effect；因为multiplicative_terms包括节假日效应、星期效应、年季节性效应，multiplicative_terms就代表综合时间效应。
sm_qielei['amt_no_effect'] = sm_qielei['amount'].values / (1+qielei_prophet_amount['total_effect']).values
# 对sm_qielei['amt_no_effect']和sm_qielei['amount']画图，比较两者的差异，横坐标为busdate, 纵坐标为amount和amt_no_effect, 用plt画图
fig = plt.figure()
plt.plot(sm_qielei['busdate'], sm_qielei['amount'], label='剔除时间效应前销量')
plt.plot(sm_qielei['busdate'], sm_qielei['amt_no_effect'], label='剔除时间效应后销量')
plt.xticks(rotation=45)
plt.xlabel('销售日期')
plt.ylabel('茄类销量')
plt.title('剔除时间效应前后，茄类销量时序对比')
plt.legend(loc='best')
plt.show()
fig.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\剔除时间效应前后，茄类销量时序对比.svg", dpi=300, bbox_inches='tight')

# 计算sm_qielei['amt_no_effect']和sm_qielei['amount']的统计信息
sm_qielei_amount_effect_compare = sm_qielei[['amount', 'amt_no_effect']].describe()
print(sm_qielei_amount_effect_compare.round(3), '\n')
sm_qielei_amount_effect_compare = sm_qielei_amount_effect_compare.round(3)
sm_qielei_amount_effect_compare.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\sm_qielei_amount_effect_compare.xlsx", sheet_name='剔除时间效应前后，历史销量的描述性统计信息对比')
# 计算sm_qielei['amt_no_effect']和sm_qielei['amount']的相关系数
print(sm_qielei[['amount', 'amt_no_effect']].corr().round(4), '\n')

# 对历史销量数据进行扩增，使更近的样本占更大的权重
sm_qielei_amt_ext = ref.extendSample(sm_qielei['amt_no_effect'].values, max_weight=int(len(sm_qielei['amt_no_effect'].values)**(extend_power)))

# 在同一个图中，画出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的分布及概率密度函数的对比图
fig, ax = plt.subplots(1, 1)
sns.distplot(sm_qielei_amt_ext, ax=ax, label='extended')
sns.distplot(sm_qielei['amt_no_effect'].values, ax=ax, label='original')
ax.legend()
plt.title('数据扩增前后，历史销量的概率密度函数对比图')
plt.show()
fig.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\\results\\数据扩增前后，历史销量的概率密度函数对比图.svg", dpi=300, bbox_inches='tight')

# 给出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的描述性统计
sm_qielei_amt_ext_describe = pd.Series(sm_qielei_amt_ext, name='sm_qielei_amt_ext_describe').describe()
sm_qielei_amt_describe = sm_qielei['amt_no_effect'].describe()
sm_qielei_amt_ext_compare = pd.concat([sm_qielei_amt_describe, sm_qielei_amt_ext_describe], axis=1).rename(columns={'amt_no_effect': 'sm_qielei_amt_describe'})
print(sm_qielei_amt_ext_compare.round(2), '\n')
sm_qielei_amt_ext_compare = sm_qielei_amt_ext_compare.round(2)
sm_qielei_amt_ext_compare.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\sm_qielei_amt_ext_compare.xlsx", sheet_name='数据扩增前后，历史销量的描述性统计信息对比')

# 给出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的Shapiro-Wilk检验结果
stat, p = stats.shapiro(sm_qielei_amt_ext)
print('"sm_qielei_amt_ext" Shapiro-Wilk test statistic:', round(stat, 4))
print('"sm_qielei_amt_ext" Shapiro-Wilk test p-value:', round(p, 4))
# Perform Shapiro-Wilk test on amt_no_effect
stat, p = stats.shapiro(sm_qielei["amt_no_effect"].values)
print('"amt_no_effect" Shapiro-Wilk test statistic:', round(stat, 4))
print('"amt_no_effect" Shapiro-Wilk test p-value:', round(p, 4), '\n')

# 计算sm_qielei在训练集上的平均毛利率
sm_qielei['price_daily_avg'] = sm_qielei['sum_price'] / sm_qielei['amount']
sm_qielei['cost_daily_avg'] = sm_qielei['sum_cost'] / sm_qielei['amount']
sm_qielei['profit_daily'] = (sm_qielei['price_daily_avg'] - sm_qielei['cost_daily_avg']) / sm_qielei['price_daily_avg']
profit_avg = (sm_qielei['profit_daily'].mean() + np.percentile(sm_qielei['profit_daily'], 50)) / 2
# 用newsvendor模型计算sm_qielei_amt_ext的平稳订货量q_steady
f = fitter.Fitter(sm_qielei_amt_ext, distributions='gamma')
f.fit()
q_steady = stats.gamma.ppf(profit_avg, *f.fitted_param['gamma'])
params = f.fitted_param["gamma"]
params_rounded = tuple(round(float(param), 4) for param in params)
print(f'拟合分布的最优参数是: \n {params_rounded}', '\n')
print(f'第一次的平稳订货量q_steady = {round(q_steady, 3)}', '\n')


# 观察sm_qielei_amt_ext的分布情况
f = fitter.Fitter(sm_qielei_amt_ext, distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'irayleigh', 'uniform'], timeout=10)
f.fit()
comparison_of_distributions_qielei = f.summary(Nbest=5)
print(f'\n{comparison_of_distributions_qielei.round(4)}\n')
comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
comparison_of_distributions_qielei.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\comparison_of_distributions_qielei.xlsx", sheet_name='comparison of distributions_qielei')

name = list(f.get_best().keys())[0]
print(f'best distribution: {name}''\n')
f.plot_pdf(Nbest=5)
figure = plt.gcf()  # 获取当前图像
plt.xlabel('用于拟合分布的，茄类数据扩增后的历史销量')
plt.ylabel('Probability')
plt.title('comparison of distributions_qielei')
plt.show()
figure.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\comparison of distributions_qielei.svg")
figure.clear()  # 先画图plt.show，再释放内存

figure = plt.gcf()  # 获取当前图像
plt.plot(f.x, f.y, 'b-.', label='f.y')
plt.plot(f.x, f.fitted_pdf[name], 'r-', label="f.fitted_pdf")
plt.xlabel('用于拟合分布的，茄类数据扩增后的历史销量')
plt.ylabel('Probability')
plt.title(f'best distribution: {name}')
plt.legend()
plt.show()
figure.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\best distribution_qielei.svg")
figure.clear()


# 将时间效应加载到q_steady上，得到预测期的最优订货量q_star
train_set = qielei_prophet_amount[['ds', 'y', 'holiday_effect', 'weekly_effect', 'yearly_effect', 'holiday', 'weekday']]
all_set = pd.merge(future_amount, train_set, on='ds', how='left')
all_set['weekday'][-periods:] = all_set['ds'][-periods:].dt.weekday # 0-6, Monday is 0
all_set['holiday'][-periods:] = all_set['ds'][-periods:].apply(lambda x: chinese_calendar.get_holiday_detail(x)[1] if chinese_calendar.get_holiday_detail(x)[0] else None)
# 计算all_set[:-periods]中，weekly_effect字段关于weekday字段每个取值的平均数，并保留ds字段
weekly_effect_avg = all_set[:-periods].groupby(['weekday'])['weekly_effect'].mean()
# 将all_set和weekly_effect_avg按weekday字段进行左连接，只保留一个weekday字段
all_set = pd.merge(all_set, weekly_effect_avg, on='weekday', how='left', suffixes=('', '_avg')).drop(columns=['weekly_effect'])
# 对预测期的节假日系数赋值
if len(set(all_set['holiday'][-periods:])) == 1:
    all_set['holiday_effect'][-periods:] = 0
else:
    raise ValueError('预测期中，存在多个节假日，需要手动设置holiday_effect')
# 提取all_set['ds']中的年、月、日
all_set['year'] = all_set['ds'].dt.year
all_set['month'] = all_set['ds'].dt.month
all_set['day'] = all_set['ds'].dt.day
# 取all_set[-periods:]和all_set[:-periods]中，月、日相同，但年不同的样本，计算年效应
yearly_effect_avg = all_set[:-periods].groupby(['month', 'day'])['yearly_effect'].mean().reset_index()
all_set = pd.merge(all_set, yearly_effect_avg, on=['month', 'day'], how='left', suffixes=('', '_avg')).drop(columns=['yearly_effect'])
# 计算q_star
q_star = q_steady * (1 + (all_set['holiday_effect'] + all_set['weekly_effect_avg'] + all_set['yearly_effect_avg'])[-periods:])
print('q_star = ', '\n', f'{round(q_star, 3)}', '\n')
all_set['y'][-periods:] = q_star
all_set['y'][:-periods] = forecast_amount['yhat'][:-periods]
all_set.drop(columns=['year', 'month', 'day'], inplace=True)
all_set.rename(columns={'y': '预测销量', 'ds': '销售日期'}, inplace=True)
all_set['训练集平均毛利率'] = profit_avg
all_set['第一次计算的平稳订货量'] = q_steady
all_set = all_set.round(3)
all_set.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\question_1_final_qielei_all_set.xlsx", index=False, encoding='utf-8-sig', sheet_name='问题1最终结果：茄类全集上的预测订货量及时间效应系数')


# 使用ref评估最后一周，即预测期的指标
sm_qielei_all = account_commodity_mean[account_commodity_mean['sm_sort_name'] == "茄类"][['busdate', 'amount']].rename(columns={'busdate': '销售日期', 'amount': '实际销量'})
sm_qielei_all = pd.merge(sm_qielei_all, all_set, on='销售日期', how='left')
sm_qielei_seg = sm_qielei_all[sm_qielei_all['销售日期'] >= str(sm_qielei_all['销售日期'].max().year)]

res = ref.regression_evaluation_single(y_true=sm_qielei_seg['实际销量'][-periods:].values, y_pred=sm_qielei_seg['预测销量'][-periods:].values)
accu_sin = ref.accuracy_single(y_true=sm_qielei_seg['实际销量'][-periods:].values, y_pred=sm_qielei_seg['预测销量'][-periods:].values)
metrics_values = [accu_sin] + list(res[:-2])
metrics_names = ['AA', 
 'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
 'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
 'MSE', 'MSLE',
 'VAR', 'R2', 'PR', 'SR', 'KT', 'WT', 'MGC']
metrics = pd.Series(data=metrics_values, index=metrics_names, name='评估指标值')
metrics = metrics.round(4)
metrics.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\qielei_metrics.xlsx", index=True, encoding='utf-8-sig', sheet_name='20种评估指标的取值')
print(f'metrics: \n {metrics}', '\n')

# 作图比较实际销量和预测销量，以及预测销量的置信区间，并输出保存图片
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(sm_qielei_seg['销售日期'][-output_index:], sm_qielei_seg['实际销量'][-output_index:], label='实际销量', marker='o')
ax.plot(sm_qielei_seg['销售日期'][-output_index:], sm_qielei_seg['预测销量'][-output_index:], label='预测销量', marker='o')
ax.fill_between(sm_qielei_seg['销售日期'][-output_index:], forecast_amount['yhat_lower'][-output_index:], forecast_amount['yhat_upper'][-output_index:], color='grey', alpha=0.2, label=f'{int(interval_width*100)}%的置信区间')
ax.set_xlabel('销售日期')
ax.set_ylabel('销量')
ax.set_title('茄类预测期第一次订货量时序对比图')
ax.legend()
plt.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\qielei_forecast.svg", dpi=300, bbox_inches='tight')
plt.show()


# question_2
qielei_prophet_price = sm_qielei[['busdate', 'sum_price']].rename(columns={'busdate': 'ds', 'sum_price': 'y'})
m_price = Prophet(seasonality_mode='multiplicative', holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=mcmc_samples, interval_width=interval_width)
m_price.add_country_holidays(country_name='CN')
m_price.add_seasonality(name='weekly', period=7, fourier_order=10, prior_scale=10)
m_price.add_seasonality(name='yearly', period=365, fourier_order=3, prior_scale=10)
m_price.fit(qielei_prophet_price)
future_price = m_price.make_future_dataframe(periods=periods)
forecast_price = m_price.predict(future_price)
fig1 = m_price.plot(forecast_price)
plt.show()
fig2 = m_price.plot_components(forecast_price)
plt.show()
fig1.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\fit_revenue.svg", dpi=300, bbox_inches='tight')
fig2.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\fit_revenue_components.svg", dpi=300, bbox_inches='tight')
forecast_price.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\销售额时序分解.xlsx", index=False, encoding='utf-8-sig', sheet_name='预测销售额')


sm_qielei['unit_cost'] = sm_qielei['sum_cost'] / sm_qielei['amount']
qielei_prophet_cost = sm_qielei[['busdate', 'unit_cost']].rename(columns={'busdate': 'ds', 'unit_cost': 'y'})
m_cost = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=mcmc_samples, interval_width=interval_width)
m_cost.add_country_holidays(country_name='CN')
m_cost.fit(qielei_prophet_cost)
future_cost = m_cost.make_future_dataframe(periods=periods)
forecast_cost = m_cost.predict(future_cost)
fig1 = m_cost.plot(forecast_cost)
plt.show()
fig2 = m_cost.plot_components(forecast_cost)
plt.show()
fig1.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\fit_unit_cost.svg", dpi=300, bbox_inches='tight')
fig2.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\fit_unit_cost_components.svg", dpi=300, bbox_inches='tight')
forecast_cost.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\单位成本时序分解.xlsx", index=False, encoding='utf-8-sig', sheet_name='预测单位成本')


forecast = forecast_price[['ds', 'yhat']][-periods:]
forecast['price'] = forecast['yhat'] / q_star
forecast['unit_cost'] = forecast_cost['yhat'][-periods:]
forecast['profit'] = (forecast['price'] - forecast['unit_cost']) / forecast['price']

# 用newsvendor模型计算sm_qielei_amt_ext的平稳订货量q_steady
f_star = fitter.Fitter(sm_qielei_amt_ext, distributions='gamma')
f_star.fit()
print(f'拟合分布的最优参数是: \n {np.array(f_star.fitted_param["gamma"]).round(4)}', '\n')
q_steady_star = []
for i in range(len(forecast['profit'])):
    q_steady_star.append(stats.gamma.ppf(forecast['profit'].values[i], *f_star.fitted_param['gamma']))
q_steady_star = np.array(q_steady_star)
print(f'q_steady_star = {q_steady_star.round(3)}', '\n')

all_set['total_effect'] = all_set[['holiday_effect', 'weekly_effect_avg', 'yearly_effect_avg']].sum(axis=1)
q_star_new = q_steady_star * (1 + all_set['total_effect'][-periods:])
print(f'q_star_new =\n{q_star_new.round(3)}', '\n')
forecast['加载毛利率时间效应的第二次报童订货量'] = q_steady_star
forecast['q_star_new'] = q_star_new
forecast.rename(columns={'ds': '销售日期', 'yhat': '预测金额', 'price': '预测单价', 'unit_cost': '预测成本单价', 'profit': '预测毛利率', 'q_star_new': '加载销量时间效应的最终订货量'}, inplace=True)
forecast[['预测金额', '预测单价', '预测成本单价']] = forecast[['预测金额', '预测单价', '预测成本单价']].applymap(lambda x: round(x, 2))
forecast[['加载毛利率时间效应的第二次报童订货量', '加载销量时间效应的最终订货量']] = forecast[['加载毛利率时间效应的第二次报童订货量', '加载销量时间效应的最终订货量']].apply(lambda x: round(x).astype(int))
forecast['预测毛利率'] = forecast['预测毛利率'].apply(lambda x: round(x, 3))
forecast_output = forecast.iloc[[output_index-1]]
forecast_output['销售日期'] = forecast_output['销售日期'].dt.date
forecast_output.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\question_2_final_qielei_forecast.xlsx", index=False, encoding='utf-8-sig', sheet_name='问题2最终结果：茄类在预测期每日的预测销售额、预测单价、预测成本、预测毛利率、加载毛利率时间效应的第二次报童订货量和加载销量时间效应的最终订货量')

# 评估指标
res_new = ref.regression_evaluation_single(y_true=sm_qielei_all['实际销量'][-periods:].values, y_pred=forecast['加载销量时间效应的最终订货量'][-periods:].values)
accu_sin_new = ref.accuracy_single(y_true=sm_qielei_all['实际销量'][-periods:].values, y_pred=forecast['加载销量时间效应的最终订货量'][-periods:].values)
metrics_values_new = [accu_sin_new] + list(res_new[:-2])
metrics_new = pd.Series(data=metrics_values_new, index=metrics_names, name='新评估指标值')
metrics_new = metrics_new.round(4)
metrics_new.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\qielei_metrics_final.xlsx", index=True, encoding='utf-8-sig', sheet_name='加载销量时间效应的最终订货量的评估指标值')
print(f'metrics_new: \n {metrics_new}', '\n')

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(sm_qielei_all['销售日期'][-output_index:], sm_qielei_all['实际销量'][-output_index:], marker='o', label='实际销量')
ax.plot(forecast['销售日期'][-output_index:], forecast['加载销量时间效应的最终订货量'][-output_index:], marker='o', label='加载销量时间效应的最终订货量')
ax.fill_between(sm_qielei_all['销售日期'][-output_index:], forecast_price['yhat_lower'][-output_index:] / forecast['预测单价'], forecast_price['yhat_upper'][-output_index:] / forecast['预测单价'], color='grey', alpha=0.2, label=f'{int(interval_width*100)}%的置信区间')
ax.set_xlabel('销售日期')
ax.set_ylabel('销量')
ax.set_title('茄类预测期加载销量时间效应的最终订货量对比图')
ax.legend()
plt.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\results\qielei_forecast_final.svg", dpi=300, bbox_inches='tight')
plt.show()

print('question_1运行完毕！')
