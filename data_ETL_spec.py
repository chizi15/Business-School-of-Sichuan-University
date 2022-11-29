import pandas as pd
pd.set_option('display.max_columns', None)


organ = 1
sort = 1
if organ == 1:
    organ = 'DH'
    sort = '鸡蛋'
else:
    organ = 'HLJ'
    if sort == 1:
        sort = '肉'
    else:
        sort = '绿叶类'

account = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\account.csv", dtype={'code': str})
print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n'
      f'{account.isnull().any()}\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n')
running = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\running.csv", dtype={'code': str})
print(f'\nrunning\n\nshape: {running.shape}\n\ndtypes:\n{running.dtypes}\n\nisnull-columns:\n'
      f'{running.isnull().any()}\n\nisnull-rows:\n{sum(running.isnull().T.any())}\n')
stock = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\stock.csv", dtype={'code': str})
print(f'\nstock\n\nshape: {stock.shape}\n\ndtypes:\n{stock.dtypes}\n\nisnull-columns:\n'
      f'{stock.isnull().any()}\n\nisnull-rows:\n{sum(stock.isnull().T.any())}\n')

match organ:
    case 'DH':
        commodity = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\commodity.csv")
        prediction = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\prediction.csv", dtype={'code': str})
        # screening commodity sheet by requirement
        comodt = commodity[commodity['name'].str.contains(f'{sort}')]
        comodt_seg = comodt.drop(labels=commodity[commodity['name'].str.contains(f'{sort}面')].index)
    case _:
        commodity = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\commodity.csv",
                                dtype={'code': str, 'sm_sort': str, 'md_sort': str, 'bg_sort': str})
        prediction = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\fresh-forecast-order.csv",
                                 dtype={'bg_sort': str, 'md_sort': str, 'sm_sort': str, 'code': str},
                                 names=['Unnamed', 'organ', 'class', 'bg_sort', 'bg_sort_name', 'md_sort', 'md_sort_name',
                                        'sm_sort', 'sm_sort_name', 'code', 'name', 'busdate', 'theory_sale', 'real_sale',
                                        'predict', 'advise_order', 'real_order'], header=0)
        promotion = pd.read_csv(f"D:\Work info\WestUnion\data\origin\\{organ}\\promotion.csv", dtype={'code': str})
        print(f'\npromotion\n\nshape: {promotion.shape}\n\ndtypes:\n{promotion.dtypes}\n\nisnull-columns:\n'
              f'{promotion.isnull().any()}\n\nisnull-rows:\n{sum(promotion.isnull().T.any())}\n')
        # screening commodity and promotion sheets by requirement
        match sort:
            case '肉':
                comodt_seg = commodity[commodity['bg_sort_name'].str.contains(f'{sort}')]
            case _:
                comodt_seg = commodity[commodity['md_sort_name'].str.contains(f'{sort}')]
        prom_seg = promotion[promotion['code'].isin(comodt_seg['code'])]

print(f'\ncommodity\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n'
      f'{commodity.isnull().any()}\n\nisnull-rows:\n{sum(commodity.isnull().T.any())}\n')
print(f'\nprediction\n\nshape: {prediction.shape}\n\ndtypes:\n{prediction.dtypes}\n\nisnull-columns:\n'
      f'{prediction.isnull().any()}\n\nisnull-rows:\n{sum(prediction.isnull().T.any())}\n')

# screening other four sheets by requirement
acct_seg = account[account['code'].isin(comodt_seg['code'])]
run_seg = running[running['code'].isin(comodt_seg['code'])]
stok_seg = stock[stock['code'].isin(comodt_seg['code'])]
pred_seg = prediction[prediction['code'].isin(comodt_seg['code'])]

# merge sheets in order
match organ:
    case 'DH':
        com_acct_seg = pd.merge(comodt_seg, acct_seg, how='left', on=['code'])
        com_acct_stk_seg = pd.merge(com_acct_seg, stok_seg, how='left', on=['organ', 'code', 'busdate'])
        com_acct_stk_seg.drop(columns=['name'], inplace=True)
        com_acct_stk_pred_seg = pd.merge(com_acct_stk_seg, pred_seg, how='outer', on=['organ', 'code', 'busdate'])
    case _:
        com_acct_seg = pd.merge(comodt_seg, acct_seg, how='left', on=['class', 'code'])
        com_acct_stk_seg = pd.merge(com_acct_seg, stok_seg, how='left', on=['organ', 'class', 'code', 'busdate'])
        com_acct_stk_seg.drop(columns=['name', 'sm_sort_name', 'md_sort_name', 'bg_sort_name'], inplace=True)
        pred_seg = pred_seg.drop(columns=['Unnamed', 'bg_sort_name', 'md_sort_name', 'sm_sort_name', 'name'])
        com_acct_stk_pred_seg = pd.merge(com_acct_stk_seg, pred_seg, how='outer',
                                         on=['organ', 'class', 'bg_sort', 'md_sort', 'sm_sort', 'code', 'busdate'])
        com_acct_stk_pred_seg = pd.merge(com_acct_stk_pred_seg, prom_seg, how='left', on=['organ', 'code', 'busdate'])

# derive merged sheet and running sheet
com_acct_stk_pred_seg.to_csv(f"D:\Work info\WestUnion\data\processed\\{organ}\\com_acct_stk_pred_{sort}.csv",
                             encoding='utf_8_sig', index=False)
run_seg.to_csv(f"D:\Work info\WestUnion\data\processed\\{organ}\\run_{sort}.csv",
               encoding='utf_8_sig', index=False)
