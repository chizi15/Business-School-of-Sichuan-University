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
import os
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
acct_grup_1 = account_1.groupby(["organ", "code"])
acct_grup_1 = acct_grup_1.filter(lambda x: x['amount'].min() > 0)
print(f"number of commodities after date filtering: {len(acct_grup_1['code'].unique())}", '\n')
# 计算acct_grup_1中每个code的平均销量，按降序排序，取出满足最小陈列量amount_min要求的code集合
acct_grup_1 = acct_grup_1.groupby(["code"]).mean().sort_values(by=['amount'], ascending=False).reset_index()
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
code_num = np.random.randint(code_num_min, code_num_max+1)
acct_grup = acct_grup.iloc[:code_num, :]

account = account[account['code'].isin(acct_grup['code'].unique())]


commodity = pd.read_csv(input_path + "commodity.csv")
# 将commodity所有字段转为str类型
commodity = commodity.astype('str')
acct_com = pd.merge(account, commodity, on=['class', 'code'], how='left')
# 若acct_com中name列中的元素，存在(k/g)，则将其替换为空字符串
acct_com['name'] = acct_com['name'].apply(lambda x: x.replace('(k/g)', ''))

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
