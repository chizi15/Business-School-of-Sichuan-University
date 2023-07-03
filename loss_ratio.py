# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 6)


input_path = r'D:\Work info\WestUnion\data\processed\HLJ\脱敏及筛选后样本数据\output'
input_path_commodity = r'D:\Work info\SCU\MathModeling\2023\data\output'
output_path = r"D:\Work info\SCU\MathModeling\2023\data\output"
first_day = '2020-04-21'
last_day = '2023-04-20'
sm_sort_name = ['食用菌', '花叶类', '水生根茎类', '辣椒类', '茄类', '花菜类']
add_order_loss = False


commodity = pd.read_excel(f'{input_path_commodity}/commodity.xlsx')
# 先转成int64，以免位数超限被转换为负数
if not isinstance(commodity['单品编码'].iloc[0], str):
    commodity[['单品编码', '小分类编码', '中分类编码', '大分类编码']] = commodity[['单品编码', '小分类编码', '中分类编码', '大分类编码']].astype('Int64').astype(str)


account = pd.read_csv(f'{input_path}/account.csv')
if not isinstance(account['code'].iloc[0], str):
    account['code'] = account['code'].astype('Int64').astype(str)
account = account[account['code'].isin(commodity['单品编码'])]
account['busdate'] = pd.to_datetime(account['busdate'], format='%Y-%m-%d')
account.sort_values(by=['busdate', 'code'], inplace=True)
# account = account[account['busdate'] >= order['busdate'].min()]
account = account[(account['busdate'] >= first_day) & (account['busdate'] <= last_day)]
account['busdate'] = account['busdate'].apply(lambda x: x.date())
print(f"account.isnull().sum():\n{account.isnull().sum()}", '\n')
account.rename(columns={'organ': '门店', 'class': '课别', 'code': '单品编码', 'busdate': '日期', 'sum_cost': '销售成本', 'unit_cost': '当天进货单价(元)'}, inplace=True)
print(account.info(), '\n')


order = pd.read_csv(f'{input_path}/订货数据.csv')
if not isinstance(order['code'].iloc[0], str):
    order['code'] = order['code'].astype('Int64').astype(str)
order = order[order['code'].isin(commodity['单品编码'])]
order.drop(columns=['order_pred', 'loss_theory'], inplace=True)
order['busdate'] = pd.to_datetime(order['busdate'])
order.sort_values(by=['busdate', 'code'], inplace=True)
order = order[(order['busdate'] >= first_day) & (order['busdate'] <= last_day)]
order['busdate'] = order['busdate'].apply(lambda x: x.date())
order = order[~order['name'].str.contains('冻')]
order = order[order['class'] != '肉课']
order.rename(columns={'class': '课别', 'code': '单品编码', 'name': '单品名称', 'busdate': '日期', 'order_real': '实际订货量', 'loss_real': '损耗率(%)'}, inplace=True)
order.drop(columns=['organ'], inplace=True)
print(f"order.isnull().sum():\n{order.isnull().sum()}", '\n')
print('order.info()','\n',order.info(), '\n')


if add_order_loss:

    account_order = pd.merge(account, order, on=['课别', '单品编码', '日期'], how='left')
    account_order = pd.merge(account_order, commodity, on=['课别', '单品编码'], how='left')
    account_order.drop(columns=['单品名称_x'], inplace=True)
    account_order.rename(columns={'单品名称_y': '单品名称'}, inplace=True)
    if account_order['单品名称'].str.contains('冻').sum() != 0:
        account_order = account_order[~account_order['单品名称'].str.contains('冻')]

    # 剔除loss_real中的异常值
    loss_real = account_order['损耗率(%)'].dropna()
    loss_real = loss_real[(loss_real <= 100) & (loss_real >= 0)]
    # 计算x轴1000个区间中，各个区间包含数目的比例
    count, division = np.histogram(loss_real, bins=1000)
    # 以division为区间分割点，以count为区间包含数目，在这1000个区间中生成len(account_order)个随机数
    random_loss_real = np.random.choice(division[:-1], size=len(account_order), p=count / len(loss_real))
    # 将两个直方图放在一张图上进行对比，并输出legend
    ax = sns.histplot(loss_real, bins=100, kde=True, alpha=0.2, label='原始损耗率分布')
    ax = sns.histplot(random_loss_real, bins=100, kde=True, alpha=0.2, label='扩充后的随机损耗率分布')
    ax.legend()
    ax.set_title('原始损耗率与扩充后的随机损耗率的分布对比')
    plt.show()
    # 将account_order中的loss_real替换为random_loss_real
    account_order['损耗率(%)'] = random_loss_real
    # 将实际订货量替换为实际销量乘以（1+损耗率）
    account_order['实际订货量'] = account_order['amount'] * (1 + account_order['损耗率(%)'] / 100)

    if not isinstance(account_order['小分类编码'].iloc[0], str):
        account_order[['单品名称', '小分类编码', '中分类编码', '大分类编码']] = account_order[['单品名称', '小分类编码', '中分类编码', '大分类编码']].astype('Int64').astype(str)
    account_order['当天进货单价(元)'] = round(account_order['当天进货单价(元)'], 2)
    account_order['损耗率(%)'] = round(account_order['损耗率(%)'], 2)

    account_order['实际订货量'] = round(account_order['实际订货量'], 0).astype('Int32').astype('Int16')
    print(f"account_order.isnull().sum():\n{account_order.isnull().sum()}", '\n')
    # 查看account_order中各字段的数据类型和位数
    print('account_order.info()','\n',account_order.info(),'\n')
    account_order.rename(columns={'amount': '销量', 'sum_price': '销售额', 'sum_disc': '折扣销售额',}, inplace=True)

    # account_order.to_excel(f'{output_path}/account_order.xlsx', index=False)
else:
    # 此处必须以commodity为基准，最后生成的loss_real_code_diff中记录数才能与data_output.py中的commodity记录数一致
    order_com = pd.merge(order, commodity, on=['课别', '单品编码', '单品名称'], how='right')

    if not isinstance(order_com['小分类编码'].iloc[0], str):
        order_com[['单品名称', '小分类编码', '中分类编码', '大分类编码']] = order_com[['单品名称', '小分类编码', '中分类编码', '大分类编码']].astype('Int64').astype(str)
    # 将order_com['损耗率(%)']中>100的值替换为100, <0的值替换为0
    order_com['损耗率(%)'] = order_com['损耗率(%)'].apply(lambda x: 100 if x > 100 else x)
    order_com['损耗率(%)'] = order_com['损耗率(%)'].apply(lambda x: 0 if x < 0 else x)
    # 计算order_com中各个单品和小分类的中位数损耗率，并忽略掉空值
    loss_real_code = order_com.groupby(['单品编码'])['损耗率(%)'].apply(lambda x: x.mean (skipna=True)).reset_index().rename(columns={'损耗率(%)': '平均损耗率(%)_单品编码'})
    loss_real_sm = order_com.groupby(['小分类编码'])['损耗率(%)'].apply(lambda x: x.mean (skipna=True)).reset_index().rename(columns={'损耗率(%)': '平均损耗率(%)_小分类编码'})
    # 填充loss_real_median中的损耗率空值
    loss_real_code['平均损耗率(%)_单品编码'].fillna(loss_real_code['平均损耗率(%)_单品编码'].mean(), inplace=True)
    loss_real_sm['平均损耗率(%)_小分类编码'].fillna(loss_real_sm['平均损耗率(%)_小分类编码'].mean(), inplace=True)
    # 将loss_real中的损耗率替换到order_com中
    order_com = pd.merge(order_com, loss_real_code, on=['单品编码'], how='left')
    order_com = pd.merge(order_com, loss_real_sm, on=['小分类编码'], how='left')
    order_com['平均损耗率(%)_单品编码'] = round(order_com['平均损耗率(%)_单品编码'], 2)
    order_com['平均损耗率(%)_小分类编码'] = round(order_com['平均损耗率(%)_小分类编码'], 2)

    print(f"order_com.isnull().sum():\n{order_com.isnull().sum()}", '\n')
    # 查看order_com中各字段的数据类型和位数
    print('order_com.info()','\n',order_com.info(),'\n')

    # order_com.to_excel(f'{output_path}/order_com.xlsx', index=False)
    
    # 打印各个不同的小分类编码和单品编码对应的损耗率
    loss_real_sm_diff = order_com.groupby(['小分类编码', '小分类名称'])['平均损耗率(%)_小分类编码'].apply(lambda x: x.unique()[0]).reset_index().rename(columns={'平均损耗率(%)_小分类编码': '平均损耗率(%)_小分类编码_不同值'})
    loss_real_code_diff = order_com.groupby(['单品编码', '单品名称'])['平均损耗率(%)_单品编码'].apply(lambda x: x.unique()[0]).reset_index().rename(columns={'平均损耗率(%)_单品编码': '平均损耗率(%)_单品编码_不同值'})
    # 按损耗率降序排列
    loss_real_sm_diff.sort_values(by=['平均损耗率(%)_小分类编码_不同值'], ascending=False, inplace=True)
    loss_real_code_diff.sort_values(by=['平均损耗率(%)_单品编码_不同值'], ascending=False, inplace=True)
    loss_real_sm_diff.reset_index(drop=True, inplace=True)
    loss_real_code_diff.reset_index(drop=True, inplace=True)
    loss_real_sm_diff.dropna(inplace=True)
    loss_real_code_diff.dropna(inplace=True)

    # 损耗表脱敏
    # 中文 转小写后 去掉 "好邻居，悦活里，悦活荟，hlj, 悦丰硕，悦令鲜" 文字
    loss_real_code_diff['单品名称'] = loss_real_code_diff['单品名称'].apply(lambda x: x.lower())
    loss_real_code_diff['单品名称'] = loss_real_code_diff['单品名称'].apply(lambda x: x.replace('hlj', ''))
    loss_real_code_diff['单品名称'] = loss_real_code_diff['单品名称'].apply(lambda x: x.replace('好邻居', ''))
    loss_real_code_diff['单品名称'] = loss_real_code_diff['单品名称'].apply(lambda x: x.replace('悦活里', ''))
    loss_real_code_diff['单品名称'] = loss_real_code_diff['单品名称'].apply(lambda x: x.replace('悦活荟', ''))
    loss_real_code_diff['单品名称'] = loss_real_code_diff['单品名称'].apply(lambda x: x.replace('悦丰硕', ''))
    loss_real_code_diff['单品名称'] = loss_real_code_diff['单品名称'].apply(lambda x: x.replace('悦令鲜', ''))

    loss_real_sm_diff['小分类名称'] = loss_real_sm_diff['小分类名称'].apply(lambda x: x.replace('好邻居', ''))
    loss_real_sm_diff['小分类名称'] = loss_real_sm_diff['小分类名称'].apply(lambda x: x.replace('悦活里', ''))
    loss_real_sm_diff['小分类名称'] = loss_real_sm_diff['小分类名称'].apply(lambda x: x.replace('悦活荟', ''))
    loss_real_sm_diff['小分类名称'] = loss_real_sm_diff['小分类名称'].apply(lambda x: x.replace('悦丰硕', ''))
    loss_real_sm_diff['小分类名称'] = loss_real_sm_diff['小分类名称'].apply(lambda x: x.replace('悦令鲜', ''))

    print(f"loss_real_sm_diff:\n{loss_real_sm_diff}", '\n')
    print(f"length of loss_real_code_diff:\n{len(loss_real_code_diff)}", '\n')
    # 将loss_real_sm_diff和loss_real_code_diff输出到同一个excel的不同sheet中
    with pd.ExcelWriter(f'{output_path}/损耗率.xlsx') as writer:
        loss_real_sm_diff.to_excel(writer, index=False, sheet_name='平均损耗率(%)_小分类编码_不同值')
        loss_real_code_diff.to_excel(writer, index=False, sheet_name='平均损耗率(%)_单品编码_不同值')
    
print("loss_ratio.py运行完毕！")
