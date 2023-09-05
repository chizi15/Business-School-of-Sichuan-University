"""
解法概述：
1. 将最近一段时间有销售的单品筛选出，按平均销量降序排序，取出满足最小陈列量要求的单品集合A。
2. 对集合A中单品使用使用qiestion_1.py中流程得到相关性筛选排序合并后的分组集合B。
3. 使用集合B中的分组，筛选出满足单品数条件的组合方式，计算各组合方式的平均毛利额，取出其值最大的组合C。
4. 对该组合C使用question_2.py中流程得到各组的最优订购量，再将最优订购量分解到各单品。
5. 使用question_2.py中流程得到各单品的最优售价，使集合B的毛利率最大。
"""

import pandas as pd
import numpy as np
import fitter
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from scipy import stats
import chinese_calendar
import matplotlib.pyplot as plt
import sys, os
base_path = os.path.dirname(os.path.dirname(__file__))  
sys.path.append(base_path)  # regression_evaluation_main所在文件夹的绝对路径
from regression_evaluation_main import regression_evaluation_def as ref
from data_output import output_path_self_use

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
periods = 1 # 预测步数
output_index = 1  # 将前output_index个预测结果作为最终结果
extend_power = 1/5 # 数据扩增的幂次
interval_width = 0.95 # prophet的置信区间宽度
first_day = '2023-06-24'
last_day = '2023-06-30'
amount_min = 2.5 # 最小陈列量
code_num_max = 30
code_num_min = 24
mcmc_samples = 0 # mcmc采样次数
coef = round(1/3, 2) # 相关系数排序分组时的阈值
corr_neg = -0.2 # 销量与售价的负相关性阈值
seasonality_mode = 'multiplicative'

distributions = ['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'irayleigh', 'uniform']
input_path = output_path_self_use + "\\"
output_path = r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3" + "\\"
if not os.path.exists(output_path):
    os.makedirs(output_path)


account = pd.read_csv(input_path + "account.csv")
account['busdate'] = pd.to_datetime(account['busdate'])
account.sort_values(by=['busdate'], inplace=True)
account['code'] = account['code'].astype('str')
acct_grup = account.groupby(["organ", "code"])
print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n{account.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_grup)}\n')
print(f"account['sum_disc'].mean(): {account['sum_disc'].mean()}")
print(f"account['sum_price'].mean(): {account['sum_price'].mean()}", '\n')
account.drop(columns=['sum_disc'], inplace=True)

# 对acount按code分组，筛选出当busdate在first_day与last_day之间时，amount均大于0的code
account_1 = account[account['busdate'].between(first_day, last_day)]
codes_between_dates = account_1.groupby('code')['busdate'].nunique()
codes_between_dates = codes_between_dates[codes_between_dates == (pd.to_datetime(last_day) - pd.to_datetime(first_day)).days + 1].index
account_1 = account_1[account_1['code'].isin(codes_between_dates)]
# 将account_1[account_1['amount'] <= 0]['code'].unique()从codes_between_dates中排除
codes_between_dates = set(codes_between_dates) - set(account_1[account_1['amount'] <= 0]['code'].unique())
account_1 = account_1[account_1['code'].isin(codes_between_dates)]
acct_grup_1 = account_1.groupby(["organ", "code"])
print(f"number of commodities after date filtering: {len(acct_grup_1['code'].unique())}", '\n')
# 计算acct_grup_1中每个code的平均销量，按降序排序，取出满足最小陈列量amount_min要求的code集合
acct_grup_1 = acct_grup_1.mean().sort_values(by=['amount'], ascending=False).reset_index()
acct_grup_1 = acct_grup_1[acct_grup_1['amount'] >= amount_min]
print(f"number of commodities after amount filtering: {len(acct_grup_1['code'].unique())}", '\n')

account = account[account['code'].isin(acct_grup_1['code'].unique())]
account['price'] = account['sum_price'] / account['amount']
account['cost_price'] = account['sum_cost'] / account['amount']
account['profit'] = (account['price'] - account['cost_price']) / account['price']
account = account.dropna()
account = account[account['profit'] > 0]
account.sort_values(by=['code', 'busdate'], inplace=True)
print(f'去除空值与非正毛利率的样本后，有{account["code"].nunique()}个codes', '\n')

# 将account中各个code的平均profit按降序排序
account_mean = account.groupby(["code"]).mean().reset_index()
account_mean['amount*profit'] = account_mean['amount'] * account_mean['profit']
acct_grup = account_mean.sort_values(by=['amount*profit'], ascending=False).reset_index()
# 从code_num_max到code_num_min中随机选择一个数
code_num = np.random.randint(min(code_num_min, acct_grup.shape[0]), min(code_num_max+1, acct_grup.shape[0]))
acct_grup = acct_grup.iloc[:code_num, :]

account = account[account['code'].isin(acct_grup['code'].unique())]


commodity = pd.read_csv(input_path + "commodity.csv")
# 将commodity所有字段转为str类型
commodity = commodity.astype('str')
acct_com = pd.merge(account, commodity, on=['class', 'code'], how='left')
# 若acct_com中name列中的元素，存在(k/g)，则将其替换为空字符串
acct_com['name'] = acct_com['name'].apply(lambda x: x.replace('(k/g)', ''))


# 第一部分：对筛选出的单品进行相关性筛选、排序、合并
# 绘制各个单品的平均销量时序图，及其分布比较，并得到最优分布
for name, data in acct_com.groupby(['name']):
    # 在acct_com中，对各个sm_sort分别画时间序列图，横坐标是busdate，纵坐标是amount
    fig = plt.figure(figsize=(20, 10))
    plt.plot(data['busdate'], data['amount'])
    plt.title(f'{name}')
    plt.show()
    fig.savefig(output_path + "单品_%s_销量时序.svg" % name)  # 按单品聚合后的平均销量和平均价格
    fig.clear()

    # 对销量序列进行分布拟合比较
    f = fitter.Fitter(data['amount'], distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f'\n{comparison_of_distributions_qielei.round(4)}\n')
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    comparison_of_distributions_qielei.to_excel(output_path + f"单品_{name}_comparison_of_distributions.xlsx", sheet_name=f'{name}_comparison of distributions')

    # 给figure添加label和title，并保存输出对比分布图
    name_dist = list(f.get_best().keys())[0]
    print(f'best distribution: {name_dist}''\n')
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f'{name}_销量分布拟合对比')
    plt.ylabel('Probability')
    plt.title(f'{name}_comparison of distributions')
    plt.show()
    figure.savefig(output_path + f"单品_{name}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    # 绘制并保存输出最优分布图
    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, 'b-.', label='f.y')
    plt.plot(f.x, f.fitted_pdf[name_dist], 'r-', label="f.fitted_pdf")
    plt.xlabel(f'{name}_销量最优分布拟合')
    plt.ylabel('Probability')
    plt.title(f'best distribution: {name_dist}')
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"单品_{name}_best distribution.svg")
    figure.clear()


# 对数变换增强正态性，以加强对相关系数计算假设条件的满足程度
acct_com['amount'] = acct_com['amount'].apply(lambda x: np.log1p(x))
acct_com['price'] = acct_com['price'].apply(lambda x: np.log1p(x))
# 筛选销量与价格负相关性强的单品
typeA = []
typeB = []
for code, data in acct_com.groupby(['name']):
    if len(data)>5:
        r = stats.spearmanr(data['amount'], data['price']).correlation
        if r < corr_neg:
            typeA.append(code)
        else:
            typeB.append(code)
# 对sale_sm['amount']和price做np.log1p的逆变换，使数据回到原来的尺度
acct_com['amount'] = acct_com['amount'].apply(lambda x: np.expm1(x))
acct_com['price'] = acct_com['price'].apply(lambda x: np.expm1(x))
sale_sm_a = acct_com[acct_com['name'].isin(typeA)]
sale_sm_b = acct_com[acct_com['name'].isin(typeB)]
print(f'销量与价格的负相关性强(小于{corr_neg})的单品一共有{sale_sm_a["name"].nunique()}个')
print(f'销量与价格的负相关性弱(大于等于{corr_neg})的单品一共有{sale_sm_b["name"].nunique()}个', '\n')
sale_sm_a.to_excel(output_path + f"单品_销售数据_销量与价格的负相关性强(小于{corr_neg})的一组.xlsx")
sale_sm_b.to_excel(output_path + f"单品_销售数据_销量与价格的负相关性弱(大于等于{corr_neg})的一组.xlsx")

# 计算负相关性强的单品序列的相关系数并画热力图。
# 先对df行转列
sale_sm_a_t = pd.pivot(sale_sm_a, index="busdate", columns="name", values="amount")
# 计算每列间的相关性
sale_sm_a_coe = sale_sm_a_t.corr(method='pearson') # Compute pairwise correlation of columns, excluding NA/null values
plt.figure(figsize=(20, 20))
sns.heatmap(sale_sm_a_coe, annot=True, xticklabels=True, yticklabels=True)
plt.savefig(output_path + "单品_销量与价格负相关性强的一组中，各个单品销量间的corr_heatmap.svg")

# 对typeA中单品按相关系数的排序进行分组
# 选择相关性大于coef的组合
groups = []
idxs = sale_sm_a_coe.index.to_list()
for idx, row in sale_sm_a_coe.iterrows():
    group = row[row > coef].index.to_list()
    groups.append(group)
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
print(f'进行相关性排序，并以相关系数大于{round(coef, 2)}为条件进行分组后的结果:\n{groups_}\n')

# 将groups_中的集合转换为列表
groups_ = [list(group) for group in groups_]
groups_.append(typeB)
print(f'最终分组结果\n{groups_}')
# 将groups_中的列表转换为df，索引为组号，列名为各个单品名
groups_df = pd.DataFrame(pd.Series(groups_), columns=['name'])
groups_df['group'] = groups_df.index+1
# 改变列的顺序
groups_df = groups_df[['group', 'name']]
groups_df.to_excel(output_path + f"单品_相关性分组结果：以相关系数大于{coef}为条件.xlsx", index=False, sheet_name='最后一组是销量对价格不敏感的，前面若干组是销量对价格敏感的')

# 对groups_中的每个组，从acct_com中筛选出对应的数据，组成list_df
list_df = [acct_com[acct_com['name'].isin(group)] for group in groups_]
# 循环对list_df中每个df按busdate进行合并groupby，并求均值
list_df_avg = [data.groupby(['busdate']).agg({'amount': 'mean', 'sum_price': 'mean', 'sum_cost': 'mean'}).reset_index() for data in list_df]
# 对list_df_avg中每个df画时间序列图，横坐标是busdate，纵坐标是amount
for i, data in enumerate(list_df_avg):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(data['busdate'], data['amount'])
    plt.title(f'{groups_[i]}')
    plt.show()
    fig.savefig(output_path + f"单品_{str(groups_[i]).replace('[', '(').replace(']', ')')}_按相关性分组合并后的销量时序.svg")  # 按单品聚合后的平均销量
    fig.clear()

    # 对销量序列进行分布拟合比较
    f = fitter.Fitter(data['amount'], distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f'\n{comparison_of_distributions_qielei.round(4)}\n')
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    # 将groups_[i]中的单品名转换为字符串，再替换异常符号，以便作为excel文件名和sheet_name表名
    groups_[i] = str(groups_[i])
    groups_[i] = groups_[i].replace('\'', '').replace('[', '(').replace(']', ')')
    comparison_of_distributions_qielei.to_excel(output_path + f"单品_{groups_[i]}_comparison_of_distributions.xlsx", sheet_name=f'{groups_[i]}_comparison of distributions')
    figure.clear()

    # 给figure添加label和title，并保存输出对比分布图
    name_dist = list(f.get_best().keys())[0]
    print(f'best distribution: {name_dist}''\n')
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f'{groups_[i]}_销量分布拟合对比')
    plt.ylabel('Probability')
    plt.title(f'{groups_[i]}_comparison of distributions')
    plt.show()
    figure.savefig(output_path + f"单品_{groups_[i]}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    # 绘制并保存输出最优分布图
    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, 'b-.', label='f.y')
    plt.plot(f.x, f.fitted_pdf[name_dist], 'r-', label="f.fitted_pdf")
    plt.xlabel(f'{groups_[i]}_销量最优分布拟合')
    plt.ylabel('Probability')
    plt.title(f'best distribution: {name_dist}')
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"单品_{groups_[i]}_best distribution.svg")
    figure.clear()



# 第二部分：计算各单品的最优订购量和最优售价
pd.set_option('display.max_rows', 20)
print(acct_com.dtypes)
pd.set_option('display.max_rows', 8)

loss_code = pd.read_excel(input_path + "损耗率.xlsx", sheet_name=1)
loss_code['单品编码'] = loss_code['单品编码'].astype('str')
print(f'\nloss_code\n\nshape: {loss_code.shape}\n\ndtypes:\n{loss_code.dtypes}\n\nisnull-columns:\n{loss_code.isnull().any()}', '\n')

acct_com = acct_com.merge(loss_code, left_on=['code', 'name'], right_on=['单品编码', '单品名称'], how='left')
acct_com['loss_rate'] = acct_com['平均损耗率(%)_单品编码_不同值'] / 100
acct_com.drop(columns=['单品编码', '单品名称', '平均损耗率(%)_单品编码_不同值'], inplace=True)
pd.set_option('display.max_rows', 20)
print(f'\nacct_com\n\nshape: {acct_com.shape}\n\ndtypes:\n{acct_com.dtypes}\n\nisnull-columns:\n{acct_com.isnull().any()}', '\n')
pd.set_option('display.max_rows', 7)

acct_com['amount'] = acct_com['amount'] * (1 + acct_com['loss_rate'])
acct_com['sum_cost'] = acct_com['amount'] * acct_com['unit_cost']
if abs(np.mean(acct_com['sum_cost'] / acct_com['amount'] - acct_com['unit_cost'])) < 1e-3:
    print('The mean of (sum_cost / amount - unit_cost) is less than 1e-3')
else:
    raise ValueError('The mean of (sum_cost / amount - unit_cost) is not less than 1e-3')
acct_com.drop(columns=['price',	'cost_price', 'profit'], inplace=True)

for _ in acct_com['name'].unique():
    sm_qielei_all = acct_com[acct_com['name'] == _]
    
    # 将全集上busdate缺失不连续的行用插值法进行填充
    sm_qielei_all_col = sm_qielei_all.set_index('busdate').resample('D').mean().interpolate(method='linear').reset_index()
    print("新增插入的行是：",'\n',pd.merge(sm_qielei_all_col, sm_qielei_all, on='busdate', how='left', indicator=True).query('_merge == "left_only"'), '\n')
    sm_qielei_all = pd.merge(sm_qielei_all_col, sm_qielei_all, on='busdate', how='left')
    sm_qielei_all.drop(columns=['amount_y', 'sum_cost_y', 'sum_price_y', 'unit_cost_y', 'loss_rate_y'], inplace=True)
    sm_qielei_all.rename(columns={'amount_x': 'amount', 'sum_cost_x': 'sum_cost', 'sum_price_x': 'sum_price', 'unit_cost_x': 'unit_cost', 'loss_rate_x': 'loss_rate'}, inplace=True)

    print(f"{_}经过差值后存在空值的字段有:\n{sm_qielei_all.columns[sm_qielei_all.isnull().sum()>0].to_list()}\n")
    pd.set_option('display.max_rows', 20)
    print(f"存在空值的字段的数据类型为：\n{sm_qielei_all[sm_qielei_all.columns[sm_qielei_all.isnull().sum()>0]].dtypes}\n")
    pd.set_option('display.max_rows', 7)
    print(f"{_}经过差值后存在空值的行数为：{sm_qielei_all.isnull().T.any().sum()}", '\n')
    # 将sm_qielei_all中存在空值的那些字段的空值用众数填充
    for col in sm_qielei_all.columns[sm_qielei_all.isnull().sum()>0]:
        sm_qielei_all[col].fillna(sm_qielei_all[col].mode()[0], inplace=True)
    print(f"{_}经过众数填充后存在空值的行数为：{sm_qielei_all.isnull().T.any().sum()}", '\n')

    sm_qielei_all = sm_qielei_all.round(3)
    sm_qielei_all.to_excel(os.path.join(output_path, f"{_}_在全集上的样本.xlsx"), index=False, sheet_name=f'{_}在全集上的样本')
    # 获取茄类在训练集上的样本
    sm_qielei = sm_qielei_all[:-periods]
    # 将训练集上非正毛利的异常样本的sum_price重新赋值
    sm_qielei.loc[sm_qielei['sum_cost'] >= sm_qielei['sum_price'], 'sum_price'] = sm_qielei.loc[sm_qielei['sum_cost'] >= sm_qielei['sum_price'], 'sum_cost'] * max(sm_qielei_all['sum_price'].mean() / sm_qielei_all['sum_cost'].mean(), 1.01)
    if sum(sm_qielei['sum_cost'] >= sm_qielei['sum_price']) > 0:
        raise ValueError('There are still some negative profit in sm_qielei')

    sm_qielei = sm_qielei.round(3)
    sm_qielei.to_excel(output_path + f"\{_}_在训练集上的样本.xlsx", index=False, sheet_name=f'{_}在训练集上的样本')


    # 对sm_qielei_all中数值型变量的列：amount，sum_cost和sum_price画时序图
    sm_qielei_all_num = sm_qielei_all.select_dtypes(include=np.number)
    for col in sm_qielei_all_num:
        print(f'{col}的时序图')
        sns.lineplot(x='busdate', y=col, data=sm_qielei_all)
        plt.xticks(rotation=45)
        plt.xlabel('busdate')
        plt.ylabel(col)
        plt.title(f'{_}Time Series Graph')
        plt.show()

    # 输出这三条时序图中，非空数据的起止日期，用循环实现
    for col in sm_qielei_all_num:
        print(f'{col}非空数据的起止日期为：{sm_qielei_all[sm_qielei_all[col].notnull()]["busdate"].min()}到{sm_qielei_all[sm_qielei_all[col].notnull()]["busdate"].max()}', '\n')

    # 断言这三个字段非空数据的起止日期相同
    assert (sm_qielei_all[sm_qielei_all['amount'].notnull()]["busdate"].min() == sm_qielei_all[sm_qielei_all['sum_cost'].notnull()]["busdate"].min() == sm_qielei_all[sm_qielei_all['sum_price'].notnull()]["busdate"].min()), "三个字段非空数据的开始日期不相同"
    assert (sm_qielei_all[sm_qielei_all['amount'].notnull()]["busdate"].max() == sm_qielei_all[sm_qielei_all['sum_cost'].notnull()]["busdate"].max() == sm_qielei_all[sm_qielei_all['sum_price'].notnull()]["busdate"].max()), "三个字段非空数据的结束日期不相同"

    # 用prophet获取训练集上的星期效应系数、节日效应系数和年季节性效应系数
    qielei_prophet_amount = sm_qielei[['busdate', 'amount']].rename(columns={'busdate': 'ds', 'amount': 'y'})
    m_amount = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode=seasonality_mode, holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=mcmc_samples, interval_width=interval_width)
    m_amount.add_country_holidays(country_name='CN')
    m_amount.fit(qielei_prophet_amount)
    future_amount = m_amount.make_future_dataframe(periods=periods)
    forecast_amount = m_amount.predict(future_amount)
    fig1 = m_amount.plot(forecast_amount)
    plt.show()
    fig2 = m_amount.plot_components(forecast_amount)
    plt.show()
    fig1.savefig(output_path + f"\{_}_fit_amount.svg", dpi=300, bbox_inches='tight')
    fig2.savefig(output_path + f"\{_}_fit_amount_components.svg", dpi=300, bbox_inches='tight')

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
    pd.set_option('display.max_rows', 20)
    print("空值填充前，各列空值数","\n",qielei_prophet_amount.isnull().sum(),'\n')
    pd.set_option('display.max_rows', 7)
    # 若qielei_prophet_amount中存在空值，则填充为该列前7天的均值
    for col in qielei_prophet_amount.columns[1:-2]:
        print(col)
        if qielei_prophet_amount[col].isnull().sum() > 0:
            qielei_prophet_amount[col] = qielei_prophet_amount[col].fillna(qielei_prophet_amount[col].rolling(7, min_periods=1).mean())
    print()
    pd.set_option('display.max_rows', 20)
    print("空值填充后，各列空值数","\n",qielei_prophet_amount.isnull().sum(),'\n')
    pd.set_option('display.max_rows', 7)
    # 保存输出带有时间效应和星期、节假日标签的茄类销量样本
    if seasonality_mode == 'multiplicative':
        # 将qielei_prophet_amount中holiday_effect','weekly_effect','yearly_effect','total_effect'四列的值都限定到某个区间
        qielei_prophet_amount[['holiday_effect','weekly_effect','yearly_effect','total_effect']] = qielei_prophet_amount[['holiday_effect','weekly_effect','yearly_effect','total_effect']].apply(lambda x: np.clip(x, -0.9, 9))
    else:
        qielei_prophet_amount[['holiday_effect','weekly_effect','yearly_effect','total_effect']] = qielei_prophet_amount[['holiday_effect','weekly_effect','yearly_effect','total_effect']].apply(lambda x: np.clip(x, np.percentile(x, 10), np.percentile(x, 90)))
    qielei_prophet_amount = qielei_prophet_amount.round(3)
    qielei_prophet_amount.to_excel(os.path.join(output_path, f"{_}_销量时序分解.xlsx"), index=False, sheet_name=f'{_}用历史销量计算出的时间效应，合并到训练集中')

    # 验证prophet分解出的各个分项的计算公式
    print(f"在乘法模式下，trend*(1+multiplicative_terms)=yhat, 即：sum(forecast['trend']*(1+forecast['multiplicative_terms'])-forecast['yhat']) = {sum(forecast_amount['trend']*(1+forecast_amount['multiplicative_terms'])-forecast_amount['yhat'])}", '\n')
    print(f"sum(forecast['multiplicative_terms']-(forecast['holidays']+forecast['weekly']+forecast['yearly'])) = {sum(forecast_amount['multiplicative_terms']-(forecast_amount['holidays']+forecast_amount['weekly']+forecast_amount['yearly']))}", '\n')

    # 剔除sm_qielei的时间相关效应，得到sm_qielei_no_effect；因为multiplicative_terms包括节假日效应、星期效应、年季节性效应，multiplicative_terms就代表综合时间效应。
    if seasonality_mode == 'multiplicative':
        sm_qielei['amt_no_effect'] = sm_qielei['amount'].values / (1+qielei_prophet_amount['total_effect']).values
    else:
        sm_qielei['amt_no_effect'] = sm_qielei['amount'].values - qielei_prophet_amount['total_effect'].values
    # 对sm_qielei['amt_no_effect']和sm_qielei['amount']画图，比较两者的差异，横坐标为busdate, 纵坐标为amount和amt_no_effect, 用plt画图
    fig = plt.figure()
    plt.plot(sm_qielei['busdate'], sm_qielei['amount'], label='剔除时间效应前销量')
    plt.plot(sm_qielei['busdate'], sm_qielei['amt_no_effect'], label='剔除时间效应后销量')
    plt.xticks(rotation=45)
    plt.xlabel('销售日期')
    plt.ylabel(f'{_}销量')
    plt.title(f'{_}剔除时间效应前后，销量时序对比')
    plt.legend(loc='best')
    plt.show()
    fig.savefig(output_path + f"\{_}_剔除时间效应前后，{_}销量时序对比.svg", dpi=300, bbox_inches='tight')

    # 计算sm_qielei['amt_no_effect']和sm_qielei['amount']的统计信息
    sm_qielei_amount_effect_compare = sm_qielei[['amount', 'amt_no_effect']].describe()
    print(sm_qielei_amount_effect_compare.round(3), '\n')
    sm_qielei_amount_effect_compare = sm_qielei_amount_effect_compare.round(3)
    sm_qielei_amount_effect_compare.to_excel(output_path + f"\{_}_剔除时间效应前后，历史销量的描述性统计信息对比.xlsx", sheet_name=f'{_}剔除时间效应前后，历史销量的描述性统计信息对比')
    # 计算sm_qielei['amt_no_effect']和sm_qielei['amount']的相关系数
    print(sm_qielei[['amount', 'amt_no_effect']].corr().round(4), '\n')

    # 对历史销量数据进行扩增，使更近的样本占更大的权重
    sm_qielei_amt_ext = ref.extendSample(sm_qielei['amt_no_effect'].values, max_weight=int(len(sm_qielei['amt_no_effect'].values)**(extend_power)))

    # 在同一个图中，画出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的分布及概率密度函数的对比图
    fig, ax = plt.subplots(1, 1)
    sns.distplot(sm_qielei_amt_ext, ax=ax, label='extended')
    sns.distplot(sm_qielei['amt_no_effect'].values, ax=ax, label='original')
    ax.legend()
    plt.title(f'{_}数据扩增前后，历史销量的概率密度函数对比图')
    plt.show()
    fig.savefig(output_path + f"\{_}_数据扩增前后，历史销量的概率密度函数对比图.svg", dpi=300, bbox_inches='tight')

    # 给出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的描述性统计
    sm_qielei_amt_ext_describe = pd.Series(sm_qielei_amt_ext, name='sm_qielei_amt_ext_describe').describe()
    sm_qielei_amt_describe = sm_qielei['amt_no_effect'].describe()
    sm_qielei_amt_ext_compare = pd.concat([sm_qielei_amt_describe, sm_qielei_amt_ext_describe], axis=1).rename(columns={'amt_no_effect': 'sm_qielei_amt_describe'})
    print(sm_qielei_amt_ext_compare.round(2), '\n')
    sm_qielei_amt_ext_compare = sm_qielei_amt_ext_compare.round(2)
    sm_qielei_amt_ext_compare.to_excel(output_path + f"\{_}_数据扩增前后，历史销量的描述性统计信息对比.xlsx", sheet_name=f'{_}数据扩增前后，历史销量的描述性统计信息对比')

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
    profit_avg = max((sm_qielei['profit_daily'].mean() + np.percentile(sm_qielei['profit_daily'], 50)) / 2, 0.01)
    # 用newsvendor模型计算sm_qielei_amt_ext的平稳订货量q_steady
    f = fitter.Fitter(sm_qielei_amt_ext, distributions='gamma')
    f.fit()
    q_steady = stats.gamma.ppf(profit_avg, *f.fitted_param['gamma'])
    params = f.fitted_param["gamma"]
    params_rounded = tuple(round(float(param), 4) for param in params)
    print(f'拟合分布的最优参数是: \n {params_rounded}', '\n')
    print(f'第一次的平稳订货量q_steady = {round(q_steady, 3)}', '\n')


    # 观察sm_qielei_amt_ext的分布情况
    f = fitter.Fitter(sm_qielei_amt_ext, distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f'\n{comparison_of_distributions_qielei.round(4)}\n')
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    comparison_of_distributions_qielei.to_excel(output_path + f"\{_}_comparison_of_distributions.xlsx", sheet_name=f'{_}_comparison of distributions')

    name = list(f.get_best().keys())[0]
    print(f'best distribution: {name}''\n')
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f'用于拟合分布的，{_}数据扩增后的历史销量')
    plt.ylabel('Probability')
    plt.title(f'{_} comparison of distributions_qielei')
    plt.show()
    figure.savefig(output_path + f"\{_}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, 'b-.', label='f.y')
    plt.plot(f.x, f.fitted_pdf[name], 'r-', label="f.fitted_pdf")
    plt.xlabel(f'用于拟合分布的，{_}数据扩增后的历史销量')
    plt.ylabel('Probability')
    plt.title(f'best distribution: {name}')
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"\{_}_best distribution.svg")
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
    all_set.to_excel(output_path + f"\{_}_全集上的第一次决策结果及时间效应系数.xlsx", index=False, encoding='utf-8-sig', sheet_name=f'问题2最终结果：{_}全集上的预测订货量及时间效应系数')


    # 使用ref评估最后一周，即预测期的指标
    sm_qielei_all = sm_qielei_all[['busdate', 'amount']].rename(columns={'busdate': '销售日期', 'amount': '实际销量'})
    sm_qielei_all = pd.merge(sm_qielei_all, all_set, on='销售日期', how='left')
    sm_qielei_seg = sm_qielei_all[sm_qielei_all['销售日期'] >= str(sm_qielei_all['销售日期'].max().year)]

    # 作图比较实际销量和预测销量，以及预测销量的置信区间，并输出保存图片
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(sm_qielei_seg['销售日期'][-output_index:], sm_qielei_seg['实际销量'][-output_index:], label='实际销量', marker='o')
    ax.plot(sm_qielei_seg['销售日期'][-output_index:], sm_qielei_seg['预测销量'][-output_index:], label='预测销量', marker='o')
    ax.fill_between(sm_qielei_seg['销售日期'][-output_index:], forecast_amount['yhat_lower'][-output_index:], forecast_amount['yhat_upper'][-output_index:], color='grey', alpha=0.2, label=f'{int(interval_width*100)}%的置信区间')
    ax.set_xlabel('销售日期')
    ax.set_ylabel('销量')
    ax.set_title(f'{_}预测期第一次订货量时序对比图')
    ax.legend()
    plt.savefig(output_path + f"\{_}_forecast.svg", dpi=300, bbox_inches='tight')
    plt.show()


    # 第二轮预测和定价
    qielei_prophet_price = sm_qielei[['busdate', 'sum_price']].rename(columns={'busdate': 'ds', 'sum_price': 'y'})
    m_price = Prophet(seasonality_mode=seasonality_mode, holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=mcmc_samples, interval_width=interval_width)
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
    fig1.savefig(output_path + f"\{_}_fit_revenue.svg", dpi=300, bbox_inches='tight')
    fig2.savefig(output_path + f"\{_}_fit_revenue_components.svg", dpi=300, bbox_inches='tight')
    # if the "yhat" column in forecast_price <= 0, then replace it with the mean of last 7 days' "yhat" before it
    forecast_price['yhat'] = forecast_price['yhat'].apply(lambda x: max(forecast_price['yhat'].rolling(7, min_periods=1).mean().iloc[-1], np.random.uniform(0, max(forecast_price['yhat'].median(), 0.1))) if x < 0 else x)

    forecast_price.to_excel(output_path + f"\{_}_销售额时序分解.xlsx", index=False, encoding='utf-8-sig', sheet_name=f'{_}预测销售额')


    sm_qielei['unit_cost'] = sm_qielei['sum_cost'] / sm_qielei['amount']
    qielei_prophet_cost = sm_qielei[['busdate', 'unit_cost']].rename(columns={'busdate': 'ds', 'unit_cost': 'y'})
    m_cost = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode=seasonality_mode, holidays_prior_scale=10, seasonality_prior_scale=10, mcmc_samples=mcmc_samples, interval_width=interval_width)
    m_cost.add_country_holidays(country_name='CN')
    m_cost.fit(qielei_prophet_cost)
    future_cost = m_cost.make_future_dataframe(periods=periods)
    forecast_cost = m_cost.predict(future_cost)
    fig1 = m_cost.plot(forecast_cost)
    plt.show()
    fig2 = m_cost.plot_components(forecast_cost)
    plt.show()
    fig1.savefig(output_path + f"\{_}_fit_unit_cost.svg", dpi=300, bbox_inches='tight')
    fig2.savefig(output_path + f"\{_}_fit_unit_cost_components.svg", dpi=300, bbox_inches='tight')
    forecast_cost.to_excel(output_path + f"\{_}_单位成本时序分解.xlsx", index=False, encoding='utf-8-sig', sheet_name=f'{_}预测单位成本')


    forecast = forecast_price[['ds', 'yhat']][-periods:]
    forecast['price'] = forecast['yhat'] / q_star
    forecast['unit_cost'] = forecast_cost['yhat'][-periods:]
    forecast['profit'] = (forecast['price'] - forecast['unit_cost']) / forecast['price']
    # 将forecast['profit']中<=0的值，替换为平均毛利率
    forecast['profit'] = forecast['profit'].apply(lambda x: max(profit_avg, 0.01) if x <= 0 else x)

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
    forecast_output = forecast.iloc[:output_index]
    forecast_output['销售日期'] = forecast_output['销售日期'].dt.date
    forecast_output.to_excel(output_path + f"\{_}_在预测期每日的预测销售额、预测单价、预测成本、预测毛利率、加载毛利率时间效应的第二次报童订货量和加载销量时间效应的最终订货量.xlsx", index=False, encoding='utf-8-sig', sheet_name=f'问题2最终结果：{_}在预测期每日的预测销售额、预测单价、预测成本、预测毛利率、加载毛利率时间效应的第二次报童订货量和加载销量时间效应的最终订货量')

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(sm_qielei_all['销售日期'][-output_index:], sm_qielei_all['实际销量'][-output_index:], marker='o', label='实际销量')
    ax.plot(forecast['销售日期'][-output_index:], forecast['加载销量时间效应的最终订货量'][-output_index:], marker='o', label='加载销量时间效应的最终订货量')
    ax.fill_between(sm_qielei_all['销售日期'][-output_index:], forecast_price['yhat_lower'][-output_index:] / forecast['预测单价'], forecast_price['yhat_upper'][-output_index:] / forecast['预测单价'], color='grey', alpha=0.2, label=f'{int(interval_width*100)}%的置信区间')
    ax.set_xlabel('销售日期')
    ax.set_ylabel('销量')
    ax.set_title(f'{_}预测期加载销量时间效应的最终订货量对比图')
    ax.legend()
    plt.savefig(output_path + f"\{_}_qielei_forecast_final.svg", dpi=300, bbox_inches='tight')
    plt.show()

    print('question_3运行完毕！', '\n\n')
    