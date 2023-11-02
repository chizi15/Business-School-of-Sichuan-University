import os
import numpy as np
import pandas as pd
from prophet import Prophet
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from data_output import output_path_self_use


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 4)
# 使plot的图中能显示中文
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei', font_scale=1)


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

# 平稳随机序列检验
def sationary_random_sequence_test(data:pd.DataFrame, name:list, name_type:str, y_name:str, significance_level:float=0.01) -> None:
    """对剔除时间类效应后的销售金额或销售量进行平稳随机序列检验
    param:
        data: 小分类或单品的df数据
        name: 小分类或单品的名称列表
        name_type: 'sm_sort_name'或'name'
        y_name: 'sum_price'或'amount
        significance_level: ADF检验的显著性水平
    """
    for i in range(len(name)):
        
        df = data[data[name_type] == name[i]][['busdate', y_name]]
        df.rename(columns={'busdate': 'ds', y_name: 'y'}, inplace=True)


        forecast = prophet_model(df, pred_periods, seasonality_mode, holidays_prior_scale, seasonality_prior_scale, holiday, weekly, yearly, monthly, quarterly, daily, weekly_fourier_order, yearly_fourier_order, monthly_fourier_order, quarterly_fourier_order, daily_fourier_order, weekly_prior_scale, yearly_prior_scale, monthly_prior_scale, quarterly_prior_scale, daily_prior_scale, weekly_period, yearly_period,  monthly_period, quarterly_period, daily_period, hourly_period, minutely_period, secondly_period, mcmc_samples, freq, interval_width)

        if seasonality_mode == 'multiplicative':
            effect = forecast[['ds', 'trend', 'multiplicative_terms']]
            df = df.merge(effect, on=['ds'], how='left')
            df['y_no_effect'] = df['y'] / df['multiplicative_terms'] - df['trend'] + df['trend'].mean()
        else:
            effect = forecast[['ds', 'trend', 'additive_terms']]
            df = df.merge(effect, on=['ds'], how='left')    
            df['y_no_effect'] = df['y'] - df['additive_terms'] - df['trend'] + df['trend'].mean()    


        # 用分位距和标准差剔除y_no_effect的离群值
        q1 = df['y_no_effect'].quantile(0.25)
        q3 = df['y_no_effect'].quantile(0.75)
        iqr = q3 - q1
        # 找到离群值
        outliers = df[(df['y_no_effect'] < (q1 - 1.5 * iqr)) | (df['y_no_effect'] > (q3 + 1.5 * iqr))]
        # 对于每一个离群值，找到附近的100个非离群值，计算他们的均值和标准差，然后生成一个随机数来代替这个离群值
        for j in outliers.index:
            # 找到附近100个点的非离群值
            non_outliers = df[(df['y_no_effect'] > (q1 - 1.5 * iqr)) & (df['y_no_effect'] < (q3 + 1.5 * iqr)) & (df.index >= j-50) & (df.index <= j+50)]
            # 计算均值和标准差
            mean = non_outliers['y_no_effect'].mean()
            std = non_outliers['y_no_effect'].std()
            # 生成一个随机数来代替这个离群值
            df.at[j, 'y_no_effect'] = np.random.normal(mean, std)


        # 将y_no_effect和y画在一张图上
        plt.figure(figsize=(20, 10))
        match y_name:
            case 'sum_price':
                plt.plot(df['ds'], df['y_no_effect'], label='去掉时间类效应的销售金额')
                plt.plot(df['ds'], df['y'], label='原始销售金额')
                # 添加x轴和y轴的标签
                plt.xlabel('销售日期')
                plt.ylabel('销售金额')
                plt.title(f'{name[i]}去掉时间类效应的销售金额和原始销售金额对比')
            case 'amount':
                plt.plot(df['ds'], df['y_no_effect'], label='去掉时间类效应的销售量')
                plt.plot(df['ds'], df['y'], label='原始销售量')
                # 添加x轴和y轴的标签
                plt.xlabel('销售日期')
                plt.ylabel('销售量')
                plt.title(f'{name[i]}去掉时间类效应的销售量和原始销售量对比')
            case _:
                raise ValueError(f"Unexpected y_name: {y_name}")

        # 显示图例
        plt.legend(loc='upper right')
        # 保存图形为SVG格式
        match name_type:
            case 'sm_sort_name':
                plt.savefig(paths[2] + f'{name[i]}_{y_name}对比.svg', format='svg')
            case 'name':
                plt.savefig(paths[2+1] + f'{name[i]}_{y_name}对比.svg', format='svg')
            case _:
                raise ValueError(f"Unexpected name_type: {name_type}")
        # 显示图形
        plt.show()


        # 画出y_no_effect的自相关图和偏自相关图
        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(211)
        sm.graphics.tsa.plot_acf(df['y_no_effect'], lags=int(len(df)-1), ax=ax1)
        ax1.set_title(f'Autocorrelation_{name[i]}')
        ax1.set_xlabel('Lags')
        ax1.set_ylabel('ACF')

        ax2 = fig.add_subplot(212)
        sm.graphics.tsa.plot_pacf(df['y_no_effect'], lags=int(len(df)/2-1), ax=ax2)
        ax2.set_title(f'Partial Autocorrelation_{name[i]}')
        ax2.set_xlabel('Lags')
        ax2.set_ylabel('PACF')

        # 调整子图之间的间距
        plt.subplots_adjust(hspace=0.4)
        match y_name:
            case 'sum_price':
                plt.suptitle(f'{name[i]}去掉时间类效应的销售金额的自相关图和偏自相关图')
            case 'amount':
                plt.suptitle(f'{name[i]}去掉时间类效应的销售量的自相关图和偏自相关图')
            case _:
                raise ValueError(f"Unexpected y_name: {y_name}")
        match name_type:
            case 'sm_sort_name':
                plt.savefig(paths[4] + f'{name[i]}_去掉时间类效应的{y_name}的自相关图和偏自相关图.svg', format='svg')
            case 'name':
                plt.savefig(paths[4+1] + f'{name[i]}_去掉时间类效应的{y_name}的自相关图和偏自相关图.svg', format='svg')
            case _:
                raise ValueError(f"Unexpected name_type: {name_type}")
        plt.show()


        # 进行ADF检验
        adf_result = sm.tsa.stattools.adfuller(df['y_no_effect'])
        print(f'adf_result: {adf_result}','\n')
        print(f'P-Value: {adf_result[1]}')
        if adf_result[1] < significance_level:
            print(f'所以在{significance_level}的显著性水平下，{y_name}平稳\n')
        else:
            print(f'所以在{significance_level}的显著性水平下，{y_name}不平稳\n')
            
        # 将adf_result转换为字符串
        adf_result_str = str(adf_result)
        # 创建一个解释性描述
        description = "ADF test result: \nThe Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.\n\nNotes: \nThe null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative that there is no unit root. \nIf the pvalue is above a critical size, then we cannot reject that there is a unit root. \nThe p-values are obtained through regression surface approximation from MacKinnon 1994, but using the updated 2010 tables. \nIf the p-value is close to significant, then the critical values should be used to judge whether to reject the null. \nThe autolag option and maxlag for it are described in Greene. \n\nReferences:\n[1] W. Green.  'Econometric Analysis,' 5th ed., Pearson, 2003. \n[2] Hamilton, J.D.  'Time Series Analysis'.  Princeton, 1994. \n[3] MacKinnon, J.G. 1994.  'Approximate asymptotic distribution functions for unit-root and cointegration tests.', `Journal of Business and Economic Statistics` 12, 167-76. \n[4] MacKinnon, J.G. 2010. 'Critical Values for Cointegration Tests.'  Queen's University, Dept of Economics, Working  Papers.  Available at http://ideas.repec.org/p/qed/wpaper/1227.html \n\nReturns: \nadf : float, The test statistic. \npvalue : float, MacKinnon's approximate p-value based on MacKinnon (1994, 2010). \nusedlag : int, The number of lags used. \nnobs : int, The number of observations used for the ADF regression and calculation of the critical values. \ncritical values : dict, Critical values for the test statistic at the 1 %, 5 %, and 10 % \\levels. Based on MacKinnon (2010). \nicbest : float, The maximized information criterion if autolag is not None. \n\n"
        # 将adf_result和解释性描述保存为文本文档
        match name_type:
            case 'sm_sort_name':
                with open(paths[6] + f'{name[i]}_{y_name}_adf_result.txt', 'w') as f:
                    f.write(description + adf_result_str)
            case 'name':
                with open(paths[6+1] + f'{name[i]}_{y_name}_adf_result.txt', 'w') as f:
                    f.write(description + adf_result_str)
            case _:
                raise ValueError(f"Unexpected name_type: {name_type} or y_name: {y_name}")


if __name__ == '__main__': 
    
    input_path = output_path_self_use
    
    script_name = os.path.basename(__file__).split('.')[0]
    base_path = "D:\\Work info\\SCU\\MathModeling\\2023\\data\\processed\\"

    output_path_sm = os.path.join(base_path, script_name, "sm")
    output_path_code = os.path.join(base_path, script_name, "code")

    paths = [output_path_sm, output_path_code, 
            os.path.join(output_path_sm, 'plot_revenue_sm\\'), 
            os.path.join(output_path_code, 'plot_revenue_code\\'), 
            os.path.join(output_path_sm, 'plot_acf_pacf_sm\\'), 
            os.path.join(output_path_code, 'plot_acf_pacf_code\\'),
            os.path.join(output_path_sm, 'adf_result\\'),
            os.path.join(output_path_code, 'adf_result\\')]

    for path in paths:
        os.makedirs(path, exist_ok=True)


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


    # 数据预处理
    code_sm = pd.read_excel(f"{input_path}" + "附件1-单品-分类.xlsx", sheet_name="Sheet1")
    code_sm[['单品编码', '分类编码']] = code_sm[['单品编码', '分类编码']].astype(str)
    print(f"code_sm['单品编码'].nunique(): {code_sm['单品编码'].nunique()}\ncode_sm['单品名称'].nunique(): {code_sm['单品名称'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\n")

    run_code = pd.read_excel(f"{input_path}" + "附件2-流水-销量-售价.xlsx", sheet_name="Sheet1")
    run_code['单品编码'] = run_code['单品编码'].astype(str)
    print(f"run_code['单品编码'].nunique(): {run_code['单品编码'].nunique()}\n")

    cost_code = pd.read_excel(f"{input_path}" + "附件3-进价-单品.xlsx", sheet_name="Sheet1")
    loss_code = pd.read_excel(f"{input_path}" + "附件4-损耗-单品.xlsx", sheet_name="Sheet1")

    code_sm.rename(columns={'单品编码': 'code', '单品名称': 'name', '分类编码': 'sm_sort', '分类名称': 'sm_sort_name'}, inplace=True)
    run_code.drop(columns=['扫码销售时间', '销售类型', '是否打折销售'], inplace=True)
    run_code.rename(columns={'单品编码': 'code', '销售日期': 'busdate', '销量(千克)': 'amount', '销售单价(元/千克)': 'price'}, inplace=True)
    cost_code.rename(columns={'单品编码': 'code', '日期': 'busdate', '批发价格(元/千克)': 'unit_cost'}, inplace=True)
    loss_code.rename(columns={'单品编码': 'code', '单品名称': 'name', '损耗率(%)': 'loss_rate'}, inplace=True)
    cost_code = cost_code.astype({'code': 'str'})
    loss_code = loss_code.astype({'code': 'str'})

    run_code['sum_price'] = run_code['amount'] * run_code['price']
    acct_code = run_code.groupby(['code', 'busdate'])[['amount', 'sum_price']].sum().reset_index()
    acct_code['price'] = acct_code['sum_price'] / acct_code['amount']

    acct_code_sm = acct_code.merge(code_sm, on='code', how='left')
    acct_code_sm_loss = acct_code_sm.merge(loss_code, on=['code', 'name'], how='left')
    acct_code_sm_loss_cost = acct_code_sm_loss.merge(cost_code, on=['code', 'busdate'], how='left')
    acct_code_sm_loss_cost['sum_cost'] = acct_code_sm_loss_cost['amount'] * acct_code_sm_loss_cost['unit_cost']
    acct_code_sm_loss_cost['profit'] = (acct_code_sm_loss_cost['sum_price'] - acct_code_sm_loss_cost['sum_cost']) / acct_code_sm_loss_cost['sum_price']
    acct_code_sm_loss_cost['price'] = acct_code_sm_loss_cost['sum_price'] / acct_code_sm_loss_cost['amount']

    acct_sm_loss_cost = acct_code_sm_loss_cost.groupby(['sm_sort', 'sm_sort_name','busdate']).agg({'amount': 'sum', 'sum_price': 'sum', 'sum_cost': 'sum', 'price': 'mean', 'loss_rate': 'mean', 'unit_cost': 'mean', 'profit': 'mean'}).reset_index()
    acct_sm_cost = acct_sm_loss_cost.drop(columns=['loss_rate'])
    acct_sm_cost.rename(columns={'unit_cost': 'cost_price'}, inplace=True)

    sale_sm = acct_sm_cost
    sale_code = acct_code_sm_loss_cost


    sm_sort_name = list(sale_sm['sm_sort_name'].value_counts().index.values)
    code_name = list(sale_code['name'].value_counts().index.values)
    print(f"sale_sm['sm_sort_name'].nunique(): {sale_sm['sm_sort_name'].nunique()}\nsale_code['name'].nunique(): {sale_code['name'].nunique()}\n")
    

    sationary_random_sequence_test(sale_sm, sm_sort_name, 'sm_sort_name', 'sum_price')
    sationary_random_sequence_test(sale_sm, sm_sort_name, 'sm_sort_name', 'amount')
    sationary_random_sequence_test(sale_code, code_name, 'name', 'sum_price')
    sationary_random_sequence_test(sale_code, code_name, 'name', 'amount')
