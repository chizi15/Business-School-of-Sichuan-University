import pandas as pd
import numpy as np
import chinese_calendar as calendar
pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 20)


process_type = 3  # 1: fundamental sheets process; 2: running; 3: forecasting and newsvendor comparison
match process_type:
    case 1:
        cv, n = 1/3, 365/1
        account = pd.read_csv("D:\Work info\WestUnion\data\origin\\account.csv",
                              parse_dates=['busdate'], infer_datetime_format=True, dtype={'code': str})
        print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n{account.isnull().any()}'
              f'\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n')
        commodity = pd.read_csv("D:\Work info\WestUnion\data\origin\\commodity.csv",
                                dtype={'code': str, 'sm_sort': str, 'md_sort': str, 'bg_sort': str})
        print(f'\ncommodity\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n'
              f'{commodity.isnull().any()}\n\nisnull-rows:\n{sum(commodity.isnull().T.any())}\n')
        stock = pd.read_csv("D:\Work info\WestUnion\data\origin\\stock.csv",
                            parse_dates=['busdate'], infer_datetime_format=True, dtype={'code': str})
        print(f'\nstock\n\nshape: {stock.shape}\n\ndtypes:\n{stock.dtypes}\n\nisnull-columns:\n'
              f'{stock.isnull().any()}\n\nisnull-rows:\n{sum(stock.isnull().T.any())}\n')

        account['weekday'] = account['busdate'].apply(lambda x: x.weekday() + 1)  # the label of Monday is 0, so +1
        df = pd.DataFrame(list(account['busdate'].apply(lambda x: calendar.get_holiday_detail(x))),
                          columns=['is_holiday', 'hol_type'])  # (True, None) is weekend, (False, 某节日)是指当天因某日调休而补班
        print(f'\ndf\n\nshape: {df.shape}\n\ndtypes:\n{df.dtypes}\n\nisnull-columns:\n{df.isnull().any()}'
              f'\n\nisnull-rows, i.e. the number of rows of non-holiday:\n{sum(df.isnull().T.any())}\n')
        if sum(df.isnull().T.any()) > 0:
            df.loc[df.isnull().T.any(), 'hol_type'] = '0'  # 将非节假日标为0
            print(f'\ndf\n\nshape: {df.shape}\n\ndtypes:\n{df.dtypes}\n\nisnull-columns:\n{df.isnull().any()}'
                  f'\n\nisnull-rows, i.e. the number of rows of non-holiday:\n{sum(df.isnull().T.any())}\n')
        account = pd.concat([account, df], axis=1)
        acct_comty = pd.merge(account, commodity, how='left', on=['class', 'code'])  # 两个df中所有相同的列必须on在一起，否则相同的列会出现多余的_x,_y
        print(f'\nacct_comty\n\nshape: {acct_comty.shape}\n\ndtypes:\n{acct_comty.dtypes}\n\nisnull-columns:\n'
              f'{acct_comty.isnull().any()}\n\nisnull-rows:\n{sum(acct_comty.isnull().T.any())}\n')
        acct_comty_stk = pd.merge(acct_comty, stock, how='left', on=['organ', 'class', 'code', 'busdate'])
        print(f'\nacct_comty_stk\n\nshape: {acct_comty_stk.shape}\n\ndtypes:\n{acct_comty_stk.dtypes}\n\nisnull-columns:\n'
              f'{acct_comty_stk.isnull().any()}\n\nisnull-rows:\n{sum(acct_comty_stk.isnull().T.any())}\n')
        if sum(acct_comty_stk.isnull().T.any()) > 0:
            acct_comty_stk.drop(index=acct_comty_stk[acct_comty_stk.isnull().T.any()].index, inplace=True)
        print(f'\nacct_comty_stk\n\nshape: {acct_comty_stk.shape}\n\ndtypes:\n{acct_comty_stk.dtypes}\n\nisnull-columns:\n'
              f'{acct_comty_stk.isnull().any()}\n\nisnull-rows:\n{sum(acct_comty_stk.isnull().T.any())}\n')
        cloumns = ['organ', 'code', 'name', 'busdate', 'weekday', 'is_holiday', 'hol_type', 'class', 'bg_sort', 'bg_sort_name',
                   'md_sort', 'md_sort_name', 'sm_sort', 'sm_sort_name', 'amount', 'sum_price', 'sum_cost', 'sum_disc',
                   'amou_stock', 'sum_stock', 'costprice']
        acct_comty_stk = acct_comty_stk[cloumns]

        truncated = 1  # 1: keep rest days; 2: delete rest days; 'no': no delete
        match truncated:
            case 1:  # keep rest days
                acct_comty_stk = acct_comty_stk[(acct_comty_stk['amou_stock'] == 0) & (acct_comty_stk['sum_disc'] == 0)]
            case 2:  # delete rest days
                acct_comty_stk = acct_comty_stk[(acct_comty_stk['amou_stock'] == 0) & (acct_comty_stk['sum_disc'] == 0) & \
                                                (~acct_comty_stk['is_holiday'])]
            case 'no':
                acct_comty_stk.index.name = 'account_original_index'
                acct_comty_stk.to_csv('D:\Work info\WestUnion\data\processed\merge-sheets-no-truncated-no-screen.csv')

        print(f'acct_comty_stk truncated {acct_comty_stk.shape}')
        codes_group = acct_comty_stk.groupby(['organ', 'code'])
        codes_cv = codes_group['amount'].agg(np.std) / codes_group['amount'].agg(np.mean)
        codes_cv.sort_values(inplace=True)
        codes_filter = codes_cv[(codes_cv > 0).values & (codes_cv <= cv).values].index.values
        print(f'经变异系数cv筛选后剩余单品数：{len(codes_filter)}')
        df = pd.DataFrame()
        for _ in range(len(codes_filter)):
            df = pd.concat([df, codes_group.get_group(codes_filter[_])])
        print(f'经变异系数cv筛选后剩余单品的历史销售总天数：{len(df)}')
        codes_filter_longer = df.groupby(['organ', 'code']).agg('count')[(df.groupby(['organ', 'code']).agg('count') >= n)['amount']].index.values
        print(f'经变异系数cv且销售天数n筛选后的单品数：{len(codes_filter_longer)}')
        account_filter = pd.DataFrame()
        for _ in range(len(codes_filter_longer)):
            account_filter = pd.concat([account_filter, acct_comty_stk[acct_comty_stk['code'] == codes_filter_longer[_]]])
        print(f'经变异系数cv且销售天数n筛选后剩余单品的历史销售总天数：{len(account_filter)}')
        account_filter.index.name = 'account_original_index'
        account_filter.to_csv(f'D:\Work info\WestUnion\data\processed\merge-sheets-truncated-{truncated}--cv-{cv:.2f}--len-{round(n)}.csv')
        print(f'acct_comty_stk truncated finally {account_filter.shape}')

    case 2:
        running = pd.read_csv("D:\Work info\WestUnion\data\origin\\running.csv",
                              parse_dates=['selldate', 'selltime'], infer_datetime_format=True, dtype={'code': str})
        print(f'\nrunning\n\nshape: {running.shape}\n\ndtypes:\n{running.dtypes}\n\nisnull-columns:\n'
              f'{running.isnull().any()}\n\nisnull-rows:\n{sum(running.isnull().T.any())}\n')

    case 3:
        account = pd.read_csv("D:\Work info\WestUnion\data\origin\\account.csv",
                              parse_dates=['busdate'], infer_datetime_format=True, dtype={'code': str})
        commodity = pd.read_csv("D:\Work info\WestUnion\data\origin\\commodity.csv",
                                dtype={'code': str, 'sm_sort': str, 'md_sort': str, 'bg_sort': str})
        stock = pd.read_csv("D:\Work info\WestUnion\data\origin\\stock.csv",
                            parse_dates=['busdate'], infer_datetime_format=True, dtype={'code': str})
        # 两个df中所有相同的列必须on在一起，否则相同的列会出现多余的_x,_y
        acct_comty = pd.merge(account, commodity, how='left', on=['class', 'code'])
        acct_comty_stk = pd.merge(acct_comty, stock, how='left', on=['organ', 'class', 'code', 'busdate'])
        if sum(acct_comty_stk.isnull().T.any()) > 0:
            acct_comty_stk.drop(index=acct_comty_stk[acct_comty_stk.isnull().T.any()].index, inplace=True)
        pred = pd.read_csv("D:\Work info\WestUnion\data\origin\\fresh-forecast-order.csv",
                           parse_dates=['busdate'], infer_datetime_format=True,
                           dtype={'bg_sort': str, 'md_sort': str, 'sm_sort': str, 'code': str},
                           names=['Unnamed', 'organ', 'class', 'bg_sort', 'bg_sort_name', 'md_sort', 'md_sort_name',
                                  'sm_sort', 'sm_sort_name', 'code', 'name', 'busdate', 'theory_sale', 'real_sale',
                                  'predict', 'advise_order', 'real_order'], header=0)
        print(f'\npred\n\nshape: {pred.shape}\n\ndtypes:\n{pred.dtypes}\n\nisnull-columns:\n'
              f'{pred.isnull().any()}\n\nisnull-rows:\n{sum(pred.isnull().T.any())}\n')
        pred.drop(columns=['Unnamed'], inplace=True)
        pred_acct_comty_stk = pd.merge(pred, acct_comty_stk, how='inner',
                                       on=['organ', 'code', 'name', 'busdate', 'class', 'bg_sort', 'bg_sort_name',
                                           'md_sort', 'md_sort_name', 'sm_sort', 'sm_sort_name'])
        print(f'\npred_acct_comty_stk\n\nshape: {pred_acct_comty_stk.shape}\n\ndtypes:\n{pred_acct_comty_stk.dtypes}'
              f'\n\nisnull-columns:\n{pred_acct_comty_stk.isnull().any()}'
              f'\n\nisnull-rows:\n{sum(pred_acct_comty_stk.isnull().T.any())}\n')
        # 将predict列中含有空值的行删除，保证所有行同时有销售值和预测值
        pred_acct_comty_stk.dropna(inplace=True, subset=['predict'])
        pred_acct_comty_stk.to_csv(f'D:\Work info\WestUnion\data\processed\process_type-{process_type}-'  
                                   f'pred_acct_comty_stk_dropna.csv', index=False)  # 注意，(f’‘)里{}外不能带:,{}内带:表示设置数值类型
        group_organ = pred_acct_comty_stk.groupby(['organ'])
        profit_organ = pd.DataFrame(round((group_organ['sum_price'].sum() - group_organ['sum_cost'].sum()) /
                                    group_organ['sum_cost'].sum() * 100, 2), columns=['毛利率(%)'])
        group_class = pred_acct_comty_stk.groupby(['organ', 'class'])
        profit_class = pd.DataFrame(round((group_class['sum_price'].sum() - group_class['sum_cost'].sum()) /
                                    group_class['sum_cost'].sum() * 100, 2), columns=['毛利率(%)'])
        group_big = pred_acct_comty_stk.groupby(['organ', 'class', 'bg_sort', 'bg_sort_name'])
        profit_big = pd.DataFrame(round((group_big['sum_price'].sum() - group_big['sum_cost'].sum()) /
                                  group_big['sum_cost'].sum() * 100, 2), columns=['毛利率(%)'])
        group_mid = pred_acct_comty_stk.groupby(['organ', 'class', 'bg_sort', 'bg_sort_name', 'md_sort', 'md_sort_name'])
        profit_mid = pd.DataFrame(round((group_mid['sum_price'].sum() - group_mid['sum_cost'].sum()) /
                                  group_mid['sum_cost'].sum() * 100, 2), columns=['毛利率(%)'])
        group_small = pred_acct_comty_stk.groupby(['organ', 'class', 'bg_sort', 'bg_sort_name', 'md_sort',
                                                   'md_sort_name', 'sm_sort', 'sm_sort_name'])
        profit_small = pd.DataFrame(round((group_small['sum_price'].sum() - group_small['sum_cost'].sum()) /
                                    group_small['sum_cost'].sum() * 100, 2), columns=['毛利率(%)'])
        group_code = pred_acct_comty_stk.groupby(['organ', 'class', 'bg_sort', 'bg_sort_name', 'md_sort', 'md_sort_name',
                                                  'sm_sort', 'sm_sort_name', 'code', 'name'])
        profit_code = pd.DataFrame(round((group_code['sum_price'].sum() - group_code['sum_cost'].sum()) /
                                   group_code['sum_cost'].sum() * 100, 2), columns=['毛利率(%)'])
        profit_all = pd.concat([profit_organ, profit_class, profit_big, profit_mid, profit_small, profit_code])
        profit_all.to_csv(f'D:\Work info\WestUnion\data\processed\profit_all.csv', encoding='utf_8_sig')
