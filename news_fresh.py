import pandas as pd
import numpy as np
import chinese_calendar as calendar
pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 20)
"""
1. 流水表中sum_sell是每笔销售中该code的实收金额，sum_disc是该code的让利金额，二者相加是该code的正价应收金额；
所以在流水表和账表中按sum_disc==0来筛选，就能得到正价的销售；所以在账表中，sum_price < sum_cost，当日该code的毛利为负，
与sum_disc的大小无关。
2. 流水表中sum_sell按code来groupby，再按日汇总，就是账表中每个code的sum_price。
3. 库存表可用来筛选每个code在哪些天无剩余库存，即当日卖完，再关联账表中的sum_disc，若也为0，则该code在当天是正价售完，可视为正价真实需求。
"""

truncated = 1
cv, n = 1, 365
account = pd.read_csv("D:\Work info\WestUnion\data\账表销售.csv", parse_dates=['busdate'])
print(f'account {account.shape}')
commodity = pd.read_csv("D:\Work info\WestUnion\data\商品资料.csv")
print(f'commodity {commodity.shape}')
# runnning = pd.read_csv("D:\Work info\WestUnion\data\销售流水.csv", parse_dates=['busdate'])
# print(f'running {running.shape}')
stock = pd.read_csv("D:\Work info\WestUnion\data\每日结存.csv", parse_dates=['busdate'])
print(f'stock {stock.shape}')

account['weekday'] = account['busdate'].apply(lambda x: x.weekday() + 1)  # the label of Monday is 0, so +1
df = pd.DataFrame(list(account['busdate'].apply(lambda x: calendar.get_holiday_detail(x))),
                  columns=['is_holiday', 'hol_type'])  # (True, None) is weekend, (False, 某节日)是指当天因某日调休而补班
account = pd.concat([account, df], axis=1)
acct_comty = pd.merge(account, commodity, how='left', on=['class', 'code'])
print(f'acct_comty {acct_comty.shape}')
acct_comty_stk = pd.merge(acct_comty, stock, how='left', on=['organ', 'class', 'code', 'busdate'])
print(f'acct_comty_stk {acct_comty_stk.shape}')
cloumns = ['organ', 'class', 'code', 'name', 'busdate', 'weekday', 'is_holiday', 'hol_type', 'amount', 'sum_price',
           'sum_cost', 'sum_disc', 'amou_stock', 'sum_stock', 'costprice']
acct_comty_stk = acct_comty_stk[cloumns]

match truncated:
    case 1:  # keep rest days
        acct_comty_stk = acct_comty_stk[(acct_comty_stk['amou_stock'] == 0) & (acct_comty_stk['sum_disc'] == 0)]
    case 2:  # delete rest days
        acct_comty_stk = acct_comty_stk[(acct_comty_stk['amou_stock'] == 0) & (acct_comty_stk['sum_disc'] == 0) & \
                                        (~acct_comty_stk['is_holiday'])]
    case _:
        acct_comty_stk.index.name = '账表原始索引'
        acct_comty_stk.to_csv('D:\Work info\WestUnion\data\合并账表未筛选.csv')

print(f'acct_comty_stk truncated {acct_comty_stk.shape}')
codes_group = acct_comty_stk.groupby(['code'])
codes_cv = codes_group['amount'].agg(np.std) / codes_group['amount'].agg(np.mean)
codes_cv.sort_values(inplace=True)
codes_filter = codes_cv[(codes_cv > 0).values & (codes_cv <= cv).values].index.values
print(f'经变异系数cv筛选后剩余单品数：{len(codes_filter)}')
df = pd.DataFrame()
for _ in range(len(codes_filter)):
    df = pd.concat([df, codes_group.get_group(codes_filter[_])])
print(f'经变异系数cv筛选后剩余单品的历史销售总天数：{len(df)}')
codes_filter_longer = df.groupby(['code']).agg('count')[(df.groupby(['code']).agg('count') >= n)['amount']].index.values
print(f'经变异系数cv且销售天数n筛选后的单品数：{len(codes_filter_longer)}')
account_filter = pd.DataFrame()
for _ in range(len(codes_filter_longer)):
    account_filter = pd.concat([account_filter, acct_comty_stk[acct_comty_stk['code'] == codes_filter_longer[_]]])
print(f'经变异系数cv且销售天数n筛选后剩余单品的历史销售总天数：{len(account_filter)}')
account_filter.index.name = '账表原始索引'
account_filter.to_csv(f'D:\Work info\WestUnion\data\合并账表筛选truncated-{truncated}--cv-{cv:.2f}--len-{n}.csv')
print(f'acct_comty_stk truncated finally {account_filter.shape}')
