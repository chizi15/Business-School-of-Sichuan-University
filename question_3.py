# -*- coding: utf-8 -*-
from prophet import Prophet
import pandas as pd
import numpy as np
from scipy import stats
import chinese_calendar
import fitter
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 两层dirname才能得到上上级目录
# 添加其他文件夹路径的脚本到系统临时路径，不会保留在环境变量中，每次重新append即可
sys.path.append(base_path)  # regression_evaluation_main所在文件夹的绝对路径
from regression_evaluation_main import regression_evaluation_def as ref
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 8)
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
extend_power = 1/5 # 数据扩增的幂次
interval_width = 0.95 # prophet的置信区间宽度

# read and summerize data
running = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\running.csv")
print(f"running['type']的取值: {running['type'].unique()}", '\n')
print(f"'退货'的最大取值: {running[running['type'] == '退货']['amount'].max()}", '\n', "退货的最大值为负数，说明退货的amount是负数, 则'退货'记录可与'单品销售'记录相加，得到每日实际销量", '\n')
run_apple = running[running['class'] == '水果课']
run_apple['busdate'] = pd.to_datetime(run_apple['selldate'], infer_datetime_format=True)
run_apple.sort_values(by=['busdate', 'selltime'], inplace=True)
run_apple = run_apple.groupby(['busdate', 'code'])[['amount', 'sum_sell', 'sum_disc']].sum().reset_index()
account_apple = run_apple.groupby(['busdate'])[['amount', 'sum_sell', 'sum_disc']].mean().reset_index()
account_apple.rename(columns={'sum_sell': 'sum_price'}, inplace=True)

account = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\account.csv")
account['busdate'] = pd.to_datetime(account['busdate'], infer_datetime_format=True)
account_seg = account[account['class'] == '水果课']
account_seg.sort_values(by=['busdate'], inplace=True)
account_seg = account_seg.groupby(['busdate'])[['sum_cost']].mean().reset_index()

account_apple = pd.merge(account_apple, account_seg, on='busdate', how='left')
account_apple.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\account_apple_processed.xlsx", index=False, sheet_name='流水表经过缺货填补，并按日聚合，再和账表合并sum_cost后的苹果日销售数据')

account_apple_train = account_apple[:-periods]
# 用prophet获取训练集上的星期效应系数、节日效应系数和年季节性效应系数
apple_prophet_amount = account_apple_train[['busdate', 'amount']].rename(columns={'busdate': 'ds', 'amount': 'y'})
m_amount = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=0, interval_width=interval_width)
m_amount.add_country_holidays(country_name='CN')
m_amount.fit(apple_prophet_amount)
future_amount = m_amount.make_future_dataframe(periods=periods)
forecast_amount = m_amount.predict(future_amount)
fig1 = m_amount.plot(forecast_amount)
plt.show()
fig2 = m_amount.plot_components(forecast_amount)
plt.show()
fig1.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_fit_amount.svg", dpi=300, bbox_inches='tight')
fig2.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_fit_amount_components.svg", dpi=300, bbox_inches='tight')

holiday_effect = forecast_amount[['ds', 'holidays', 'holidays_lower', 'holidays_upper']]
weekly_effect = forecast_amount[['ds', 'weekly', 'weekly_lower', 'weekly_upper']]
yearly_effect = forecast_amount[['ds', 'yearly', 'yearly_lower', 'yearly_upper']]
multiplicative_terms = forecast_amount[['ds', 'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper']]

# 根据apple_prophet_amount的ds列，将holiday_effect, weekly_effect, yearly_effect, multiplicative_terms左连接合并到apple_prophet_amount中
apple_prophet_amount = pd.merge(apple_prophet_amount, holiday_effect, on='ds', how='left')
apple_prophet_amount = pd.merge(apple_prophet_amount, weekly_effect, on='ds', how='left')
apple_prophet_amount = pd.merge(apple_prophet_amount, yearly_effect, on='ds', how='left')
apple_prophet_amount = pd.merge(apple_prophet_amount, multiplicative_terms, on='ds', how='left')
apple_prophet_amount = apple_prophet_amount.rename(columns={'holidays': 'holiday_effect', 'weekly': 'weekly_effect', 'yearly': 'yearly_effect', 'multiplicative_terms': 'total_effect'})
print(apple_prophet_amount.isnull().sum(), '\n')

# 在apple_prophet_amount中，增加中国日历的星期和节假日的列，列名分别为weekday和holiday
apple_prophet_amount['weekday'] = apple_prophet_amount['ds'].dt.weekday # 0-6, Monday is 0
# 根据日期获取中国节日名称，使用chinese_calendar库
apple_prophet_amount['holiday'] = apple_prophet_amount['ds'].apply(lambda x: chinese_calendar.get_holiday_detail(x)[1] if chinese_calendar.get_holiday_detail(x)[0] else None)

# 保存输出带有时间效应和星期、节假日标签的苹果销量样本
apple_prophet_amount.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_prophet_amount_with_effect.xlsx", index=False, sheet_name='用苹果历史销量计算出的时间效应，合并到训练集中')

# 验证prophet分解出的各个分项的计算公式
print(f"在乘法模式下，trend*(1+multiplicative_terms)=yhat, 即：sum(forecast['trend']*(1+forecast['multiplicative_terms'])-forecast['yhat']) = {sum(forecast_amount['trend']*(1+forecast_amount['multiplicative_terms'])-forecast_amount['yhat'])}", '\n')
print(f"sum(forecast['multiplicative_terms']-(forecast['holidays']+forecast['weekly']+forecast['yearly'])) = {sum(forecast_amount['multiplicative_terms']-(forecast_amount['holidays']+forecast_amount['weekly']+forecast_amount['yearly']))}", '\n')


account_apple_train['amt_no_effect'] = account_apple_train['amount'].values / (1+apple_prophet_amount['total_effect']).values

fig = plt.figure()
plt.plot(account_apple_train['busdate'], account_apple_train['amount'], label='剔除时间效应前销量')
plt.plot(account_apple_train['busdate'], account_apple_train['amt_no_effect'], label='剔除时间效应后销量')
plt.xticks(rotation=45)
plt.xlabel('销售日期')
plt.ylabel('苹果销量')
plt.title('剔除时间效应前后，苹果销量时序对比')
plt.legend(loc='best')
plt.show()
fig.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\剔除时间效应前后，苹果销量时序对比.svg", dpi=300, bbox_inches='tight')

# 计算剔除时间效应前后，苹果历史销量的描述性统计信息对比
apple_train_amount_effect_compare = account_apple_train[['amount', 'amt_no_effect']].describe()
print(apple_train_amount_effect_compare)
apple_train_amount_effect_compare.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_train_amount_effect_compare.xlsx", sheet_name='剔除时间效应前后，苹果历史销量的描述性统计信息对比')

print(account_apple_train[['amount', 'amt_no_effect']].corr(), '\n')

# 对历史销量数据进行扩增，使更近的样本占更大的权重
apple_train_amt_ext = ref.extendSample(account_apple_train['amt_no_effect'].values, max_weight=int(len(account_apple_train['amt_no_effect'].values)**(extend_power)))

# 绘制数据扩增前后，苹果历史销量的概率密度函数对比图
fig, ax = plt.subplots(1, 1)
sns.distplot(apple_train_amt_ext, ax=ax, label='extended')
sns.distplot(account_apple_train['amt_no_effect'].values, ax=ax, label='original')
ax.legend()
plt.title('数据扩增前后，苹果历史销量的概率密度函数对比图')
plt.show()
fig.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1_2\数据扩增前后，历史销量的概率密度函数对比图.svg", dpi=300, bbox_inches='tight')

# 给出数据扩增前后，苹果历史销量的描述性统计信息对比
apple_train_amt_ext_describe = pd.Series(apple_train_amt_ext, name='apple_train_amt_ext_describe').describe()
apple_train_amt_describe = account_apple_train['amt_no_effect'].describe()
apple_train_amt_ext_compare = pd.concat([apple_train_amt_describe, apple_train_amt_ext_describe], axis=1).rename(columns={'amt_no_effect': 'apple_train_amt_describe'})
print(apple_train_amt_ext_compare, '\n')
apple_train_amt_ext_compare.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_train_amt_ext_compare.xlsx", sheet_name='数据扩增前后，苹果历史销量的描述性统计信息对比')

# 对数据扩增前后，苹果历史销量进行正态性检验
stat, p = stats.shapiro(apple_train_amt_ext)
print('"apple_train_amt_ext" Shapiro-Wilk test statistic:', stat)
print('"apple_train_amt_ext" Shapiro-Wilk test p-value:', p)
# Perform Shapiro-Wilk test on amt_no_effect
stat, p = stats.shapiro(account_apple_train["amt_no_effect"].values)
print('"amt_no_effect" Shapiro-Wilk test statistic:', stat)
print('"amt_no_effect" Shapiro-Wilk test p-value:', p, '\n')


account_apple_train['price_daily_avg'] = account_apple_train['sum_price'] / account_apple_train['amount']
account_apple_train['cost_daily_avg'] = account_apple_train['sum_cost'] / account_apple_train['amount']
account_apple_train['profit_daily'] = (account_apple_train['price_daily_avg'] - account_apple_train['cost_daily_avg']) / account_apple_train['price_daily_avg']
profit_avg = (account_apple_train['profit_daily'].mean() + np.percentile(account_apple_train['profit_daily'], 50)) / 2

f = fitter.Fitter(apple_train_amt_ext, distributions='gamma')
f.fit()
q_steady = stats.gamma.ppf(profit_avg, *f.fitted_param['gamma'])
print(f'拟合分布的最优参数是: \n {f.fitted_param["gamma"]}', '\n')
print(f'q_steady = {q_steady}', '\n')


# 观察apple_train_amt_ext的分布情况
f = fitter.Fitter(apple_train_amt_ext, distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'irayleigh', 'uniform'], timeout=10)
f.fit()
comparison_of_distributions_apple = f.summary(Nbest=5)
print(f'\n{comparison_of_distributions_apple}\n')
comparison_of_distributions_apple.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\comparison_of_distributions_apple.xlsx", sheet_name='comparison of apple distributions')

name = list(f.get_best().keys())[0]
print(f'best distribution: {name}''\n')
f.plot_pdf(Nbest=5)
figure = plt.gcf()  # 获取当前图像
plt.xlabel('用于拟合分布的，苹果数据扩增后的历史销量')
plt.ylabel('Probability')
plt.title('comparison of apple distributions')
plt.show()
figure.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\comparison of distributions_apple.svg")
figure.clear()  # 先画图plt.show，再释放内存

figure = plt.gcf()  # 获取当前图像
plt.plot(f.x, f.y, 'b-.', label='f.y')
plt.plot(f.x, f.fitted_pdf[name], 'r-', label="f.fitted_pdf")
plt.xlabel('用于拟合分布的，茄类数据扩增后的历史销量')
plt.ylabel('Probability')
plt.title(f'best distribution: {name}')
plt.legend()
plt.show()
figure.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\best distribution_apple.svg")
figure.clear()


# 将时间效应加载到q_steady上，得到预测期的最优订货量q_star
train_set = apple_prophet_amount[['ds', 'y', 'holiday_effect', 'weekly_effect', 'yearly_effect', 'holiday', 'weekday']]
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
print('q_star = ', '\n', f'{q_star}', '\n')
all_set['y'][-periods:] = q_star
all_set['y'][:-periods] = forecast_amount['yhat'][:-periods]
all_set.drop(columns=['year', 'month', 'day'], inplace=True)
all_set.rename(columns={'y': '预测销量', 'ds': '销售日期'}, inplace=True)
all_set['训练集平均毛利率'] = profit_avg
all_set['第一次计算的平稳订货量'] = q_steady
all_set.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_all_set.xlsx", index=False, encoding='utf-8-sig', sheet_name='苹果在全集上的第一次预测订货量及时间效应系数')


# 使用ref评估最后一周，即预测期的指标
apple_all = account_apple[['busdate', 'amount']].rename(columns={'busdate': '销售日期', 'amount': '实际销量'})
apple_all = pd.merge(apple_all, all_set, on='销售日期', how='left')

res = ref.regression_evaluation_single(y_true=apple_all['实际销量'][-periods:].values, y_pred=apple_all['预测销量'][-periods:].values)
accu_sin = ref.accuracy_single(y_true=apple_all['实际销量'][-periods:].values, y_pred=apple_all['预测销量'][-periods:].values)
metrics_values = [accu_sin] + list(res[:-2])
metrics_names = ['AA', 
 'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
 'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
 'MSE', 'MSLE',
 'VAR', 'R2', 'PR', 'SR', 'KT', 'WT', 'MGC']
metrics = pd.Series(data=metrics_values, index=metrics_names, name='评估指标值')
metrics.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_metrics.xlsx", index=True, encoding='utf-8-sig', sheet_name='第一次计算订货量时，20种评估指标的取值')
print(f'metrics: \n {metrics}', '\n')

# 作图比较实际销量和预测销量，以及预测销量的置信区间，并输出保存图片
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(apple_all['销售日期'][-periods:], apple_all['实际销量'][-periods:], label='实际销量')
ax.plot(apple_all['销售日期'][-periods:], apple_all['预测销量'][-periods:], label='预测销量')
ax.fill_between(apple_all['销售日期'][-periods:], forecast_amount['yhat_lower'][-periods:], forecast_amount['yhat_upper'][-periods:], color='grey', alpha=0.2, label=f'{int(interval_width*100)}%的置信区间')
ax.set_xlabel('销售日期')
ax.set_ylabel('销量')
ax.set_title('苹果预测期第一次订货量时序对比图')
ax.legend()
plt.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_forecast.svg", dpi=300, bbox_inches='tight')
plt.show()


# question_2
apple_prophet_price = account_apple_train[['busdate', 'sum_price']].rename(columns={'busdate': 'ds', 'sum_price': 'y'})
m_price = Prophet(seasonality_mode='multiplicative', holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=0, interval_width=interval_width)
m_price.add_country_holidays(country_name='CN')
m_price.add_seasonality(name='weekly', period=7, fourier_order=10, prior_scale=10)
m_price.add_seasonality(name='yearly', period=365, fourier_order=3, prior_scale=10)
m_price.add_seasonality(name='monthly', period=30.5, fourier_order=2, prior_scale=1)
m_price.add_seasonality(name='quarterly', period=91.25, fourier_order=2, prior_scale=1)
m_price.fit(apple_prophet_price)
future_price = m_price.make_future_dataframe(periods=periods)
forecast_price = m_price.predict(future_price)
fig1 = m_price.plot(forecast_price)
plt.show()
fig2 = m_price.plot_components(forecast_price)
plt.show()
fig1.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_fit_price.svg", dpi=300, bbox_inches='tight')
fig2.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_fit_price_components.svg", dpi=300, bbox_inches='tight')

account_apple_train['cost'] = account_apple_train['sum_cost'] / account_apple_train['amount']
apple_prophet_cost = account_apple_train[['busdate', 'cost']].rename(columns={'busdate': 'ds', 'cost': 'y'})
m_cost = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive', holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=0, interval_width=interval_width)
m_cost.add_country_holidays(country_name='CN')
m_cost.fit(apple_prophet_cost)
future_cost = m_cost.make_future_dataframe(periods=periods)
forecast_cost = m_cost.predict(future_cost)
fig1 = m_cost.plot(forecast_cost)
plt.show()
fig2 = m_cost.plot_components(forecast_cost)
plt.show()
fig1.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_fit_cost.svg", dpi=300, bbox_inches='tight')
fig2.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_fit_cost_components.svg", dpi=300, bbox_inches='tight')

forecast = forecast_price[['ds', 'yhat']][-periods:]
forecast['price'] = forecast['yhat'] / q_star
forecast['cost'] = forecast_cost['yhat'][-periods:]
forecast['profit'] = (forecast['price'] - forecast['cost']) / forecast['price']
profit_mean = forecast['profit'].mean()

f_star = fitter.Fitter(apple_train_amt_ext, distributions='gamma')
f_star.fit()
print(f'拟合分布的最优参数是: \n {f_star.fitted_param["gamma"]}', '\n')
q_steady_star = []
for i in range(len(forecast['profit'])):
    q = stats.gamma.ppf(forecast['profit'].values[i], *f_star.fitted_param['gamma'])
    if math.isnan(q):
        print(math.isnan(q))
        q_steady_star.append((np.mean(account_apple_train['amt_no_effect'].values[-periods:]) + np.percentile(account_apple_train['amt_no_effect'].values[-periods:], 50)) / 2)
    else:
        q_steady_star.append(stats.gamma.ppf(forecast['profit'].values[i], *f_star.fitted_param['gamma']))
q_steady_star = np.array(q_steady_star)
print(f'q_steady_star = {q_steady_star}', '\n')

all_set['total_effect'] = all_set[['holiday_effect', 'weekly_effect_avg', 'yearly_effect_avg']].sum(axis=1)
q_star_new = q_steady_star * (1 + all_set['total_effect'][-periods:])
forecast['未加载时间效应的第二次报童订货量'] = q_steady_star
forecast['q_star_new'] = q_star_new
forecast.rename(columns={'ds': '销售日期', 'yhat': '预测金额', 'price': '预测单价', 'cost': '预测成本', 'profit': '预测毛利率', 'q_star_new': '新订货量'}, inplace=True)
# 将forecast中预测毛利率小于0的元素替换为其他预测毛利率的均值
forecast['预测毛利率'] = forecast['预测毛利率'].apply(lambda x: profit_mean if x < 0 else x)
forecast['预测成本'] = forecast['预测单价'] - forecast['预测毛利率'] * forecast['预测单价']
forecast.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\question_5_final_apple_forecast.xlsx", index=False, encoding='utf-8-sig', sheet_name='问题5最终结果：苹果在预测期每日的预测销售额、预测单价、预测成本、预测毛利率、未加载时间效应的第二次报童订货量和新订货量')

# 评估指标
res_new = ref.regression_evaluation_single(y_true=apple_all['实际销量'][-periods:].values, y_pred=forecast['新订货量'][-periods:].values)
accu_sin_new = ref.accuracy_single(y_true=apple_all['实际销量'][-periods:].values, y_pred=forecast['新订货量'][-periods:].values)
metrics_values_new = [accu_sin_new] + list(res_new[:-2])
metrics_new = pd.Series(data=metrics_values_new, index=metrics_names, name='新评估指标值')
metrics_new.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_metrics_new.xlsx", index=True, encoding='utf-8-sig', sheet_name='新订货量的评估指标值')
print(f'metrics_new: \n {metrics_new}', '\n')

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(apple_all['销售日期'][-periods:], apple_all['实际销量'][-periods:], label='实际销量')
ax.plot(forecast['销售日期'][-periods:], forecast['新订货量'][-periods:], label='新订货量')
ax.fill_between(apple_all['销售日期'][-periods:], forecast_price['yhat_lower'][-periods:] / forecast['预测单价'], forecast_price['yhat_upper'][-periods:] / forecast['预测单价'], color='grey', alpha=0.2, label=f'{int(interval_width*100)}%的置信区间')
ax.set_xlabel('销售日期')
ax.set_ylabel('销量')
ax.set_title('苹果预测期新订货量对比图')
ax.legend()
plt.savefig(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\results\apple_forecast_new.svg", dpi=300, bbox_inches='tight')
plt.show()
