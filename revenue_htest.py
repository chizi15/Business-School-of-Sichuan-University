import pandas as pd
from prophet import Prophet
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import sys, os
base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)
from regression_evaluation_main import regression_evaluation_def as ref
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 4)


# 模型参数
seasonality_mode = tuple(['additive', 'multiplicative'])[0]
holiday = True
weekly = True
yearly = True
monthly = True
quarterly = True
daily = False
hourly = False
minutely = False
secondly = False

holidays_prior_scale=100
seasonality_prior_scale=10
weekly_prior_scale = 100
yearly_prior_scale = 100
monthly_prior_scale = 10
quarterly_prior_scale = 10
daily_prior_scale = 1
hourly_prior_scale = 1
minutely_prior_scale = 1
secondly_prior_scale = 1

weekly_fourier_order = 10
yearly_fourier_order = 20
monthly_fourier_order = 3
quarterly_fourier_order = 3
daily_fourier_order = 2
hourly_fourier_order = 2
minutely_fourier_order = 2
secondly_fourier_order = 2

mcmc_samples = 0
interval_width = 0.95
pred_periods = 7
freq = '1D'
# 提取freq中代表数字的字符串，将其转换为数字
frequency = int(''.join([x for x in freq if x.isdigit()]))

weekly_period = frequency*7
yearly_period = frequency*365
monthly_period = frequency*30.5
quarterly_period = frequency*91.25
daily_period = frequency*1
hourly_period = frequency/24
minutely_period = frequency/24/60
secondly_period = frequency/24/60/60


def prophet_model(df, pred_periods, seasonality_mode, holidays_prior_scale, seasonality_prior_scale, holiday, weekly, yearly, monthly, quarterly, daily, weekly_fourier_order, yearly_fourier_order, monthly_fourier_order, quarterly_fourier_order, daily_fourier_order, weekly_prior_scale, yearly_prior_scale, monthly_prior_scale, quarterly_prior_scale, daily_prior_scale, weekly_period, yearly_period,  monthly_period, quarterly_period, daily_period, hourly_period, minutely_period, secondly_period, mcmc_samples, freq, interval_width):
    m = Prophet(seasonality_mode=seasonality_mode, holidays_prior_scale=holidays_prior_scale, seasonality_prior_scale=seasonality_prior_scale, mcmc_samples=mcmc_samples, interval_width=interval_width)
    if holiday:
        m.add_country_holidays(country_name='CN')
    if weekly:
        m.add_seasonality(name='weekly', period=weekly_period, fourier_order=weekly_fourier_order, prior_scale=weekly_prior_scale)
    if yearly:
        m.add_seasonality(name='yearly', period=yearly_period, fourier_order=yearly_fourier_order, prior_scale=yearly_prior_scale)
    if monthly:
        m.add_seasonality(name='monthly', period=monthly_period, fourier_order=monthly_fourier_order, prior_scale=monthly_prior_scale)
    if quarterly:
        m.add_seasonality(name='quarterly', period=quarterly_period, fourier_order=quarterly_fourier_order, prior_scale=quarterly_prior_scale)
    if daily:
        m.add_seasonality(name='daily', period=daily_period, fourier_order=daily_fourier_order, prior_scale=daily_prior_scale)
    if hourly:
        m.add_seasonality(name='hourly', period=hourly_period, fourier_order=hourly_fourier_order, prior_scale=hourly_prior_scale)
    if minutely:
        m.add_seasonality(name='minutely', period=minutely_period, fourier_order=minutely_fourier_order, prior_scale=minutely_prior_scale)
    if secondly:
        m.add_seasonality(name='secondly', period=secondly_period, fourier_order=secondly_fourier_order, prior_scale=secondly_prior_scale)
    m.fit(df)
    future = m.make_future_dataframe(periods=pred_periods, freq=freq)
    forecast = m.predict(future)
    fig1 = m.plot(forecast, uncertainty=True)
    plt.show()
    fig2 = m.plot_components(forecast, uncertainty=True)
    plt.show()
    # 保存模型
    with open('model.pkl', 'wb') as f:
        pickle.dump(m, f)

    return forecast


if __name__ == '__main__':
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


    account_commodity = pd.merge(account, commodity, on=['class', 'code'], how='left')
    print(f'\naccount_commodity\n\nshape: {account_commodity.shape}\n\ndtypes:\n{account_commodity.dtypes}\n\nisnull-columns:\n{account_commodity.isnull().any()}', '\n')


    df = account_commodity[['busdate', 'sum_price', 'sm_sort_name']]
    df = df[df['sm_sort_name'] == '茄类']
    # 将df按照busdate进行分组聚合，取均值
    df = df.groupby(['busdate']).mean().reset_index()
    df.rename(columns={'busdate': 'ds', 'sum_price': 'y'}, inplace=True)


    forecast = prophet_model(df, pred_periods, seasonality_mode, holidays_prior_scale, seasonality_prior_scale, holiday, weekly, yearly, monthly, quarterly, daily, weekly_fourier_order, yearly_fourier_order, monthly_fourier_order, quarterly_fourier_order, daily_fourier_order, weekly_prior_scale, yearly_prior_scale, monthly_prior_scale, quarterly_prior_scale, daily_prior_scale, weekly_period, yearly_period,  monthly_period, quarterly_period, daily_period, hourly_period, minutely_period, secondly_period, mcmc_samples, freq, interval_width)


    if seasonality_mode == 'multiplicative':
        effect = forecast[['ds', 'trend', 'multiplicative_terms']]
        df = df.merge(effect, on=['ds'], how='left')
        df['revenue_no_effect'] = df['y'] / df['multiplicative_terms'] - df['trend'] + df['trend'].mean()
    else:
        effect = forecast[['ds', 'trend', 'additive_terms']]
        df = df.merge(effect, on=['ds'], how='left')    
        df['revenue_no_effect'] = df['y'] - df['additive_terms'] - df['trend'] + df['trend'].mean()
    # 用分位距和标准差剔除revenue_no_effect的离群值
    q1 = df['revenue_no_effect'].quantile(0.25)
    q3 = df['revenue_no_effect'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['revenue_no_effect'] > (q1 - 1.5 * iqr)) & (df['revenue_no_effect'] < (q3 + 1.5 * iqr))]

    # 将y和revenue_no_effect画在一张图上
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df['ds'], df['y'], label='y')
    ax.plot(df['ds'], df['revenue_no_effect'], label='revenue_no_effect')
    ax.legend()
    plt.show()

    # 画出序列的自相关图和偏自相关图
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df['revenue_no_effect'], lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df['revenue_no_effect'], lags=40, ax=ax2)
    plt.show()

    # 并进行ADF检验
    adf_result = sm.tsa.stattools.adfuller(df['revenue_no_effect'])
    print(f'adf_result: {adf_result}','\n')
    print(f'adf_result[1]: {adf_result[1]}','\n')
    if adf_result[1] < 0.01:
        print('revenue_no_effect平稳')
    else:
        print('revenue_no_effect不平稳')
    