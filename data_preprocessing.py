import pandas as pd
pd.set_option('display.max_columns', None)


data = 0
if data == 0:
    account = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\\account.csv", dtype={'code': str})
    print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n'
          f'{account.isnull().any()}\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n')
    running = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\\running.csv", dtype={'code': str})
    print(f'\nrunning\n\nshape: {running.shape}\n\ndtypes:\n{running.dtypes}\n\nisnull-columns:\n'
          f'{running.isnull().any()}\n\nisnull-rows:\n{sum(running.isnull().T.any())}\n')
    commodity = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\\commodity.csv")
    print(f'\ncommodity\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n'
          f'{commodity.isnull().any()}\n\nisnull-rows:\n{sum(commodity.isnull().T.any())}\n')
    stock = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\\stock.csv", dtype={'code': str})
    print(f'\nstock\n\nshape: {stock.shape}\n\ndtypes:\n{stock.dtypes}\n\nisnull-columns:\n'
          f'{stock.isnull().any()}\n\nisnull-rows:\n{sum(stock.isnull().T.any())}\n')
    prediction = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\\prediction.csv", dtype={'code': str})
    print(f'\nprediction\n\nshape: {prediction.shape}\n\ndtypes:\n{prediction.dtypes}\n\nisnull-columns:\n'
          f'{prediction.isnull().any()}\n\nisnull-rows:\n{sum(prediction.isnull().T.any())}\n')

    comodt_egg_all = commodity[commodity['name'].str.contains('鸡蛋')]
    comodt_egg = comodt_egg_all.drop(labels=commodity[commodity['name'].str.contains('鸡蛋面')].index)
    run_egg = running[running['code'].isin(comodt_egg['code'])]
    acct_egg = account[account['code'].isin(comodt_egg['code'])]
    pred_egg = prediction[prediction['code'].isin(comodt_egg['code'])]
    stok_egg = stock[stock['code'].isin(comodt_egg['code'])]
    acct_pred_egg = pd.merge(acct_egg, pred_egg, how='outer', on=['organ', 'code', 'busdate'])
    acct_pred_com_egg = pd.merge(acct_pred_egg, comodt_egg, how='right', on=['code'])
    acct_pred_com_stk_egg = pd.merge(acct_pred_com_egg, stok_egg, how='left', on=['organ', 'code', 'busdate'])
    acct_pred_com_stk_egg.drop(columns=['name'], inplace=True)
    acct_pred_com_stk_egg.to_csv("D:\Work info\WestUnion\data\processed\dehui\\acct_pred_com_stk_egg.csv",
                                 encoding='utf_8_sig', index=False)
    run_egg.to_csv("D:\Work info\WestUnion\data\processed\dehui\\run_egg.csv", index=False)

else:
    account = pd.read_csv("D:\Work info\WestUnion\data\origin\haolinju\\account.csv", dtype={'code': str})
    print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n{account.isnull().any()}'
          f'\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n')
    running = pd.read_csv("D:\Work info\WestUnion\data\origin\haolinju\\running.csv", dtype={'code': str})
    print(f'\nrunning\n\nshape: {running.shape}\n\ndtypes:\n{running.dtypes}\n\nisnull-columns:\n'
          f'{running.isnull().any()}\n\nisnull-rows:\n{sum(running.isnull().T.any())}\n')
    commodity = pd.read_csv("D:\Work info\WestUnion\data\origin\haolinju\\commodity.csv",
                            dtype={'code': str, 'sm_sort': str, 'md_sort': str, 'bg_sort': str})
    print(f'\ncommodity\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n'
          f'{commodity.isnull().any()}\n\nisnull-rows:\n{sum(commodity.isnull().T.any())}\n')
    stock = pd.read_csv("D:\Work info\WestUnion\data\origin\haolinju\\stock.csv", dtype={'code': str})
    print(f'\nstock\n\nshape: {stock.shape}\n\ndtypes:\n{stock.dtypes}\n\nisnull-columns:\n'
          f'{stock.isnull().any()}\n\nisnull-rows:\n{sum(stock.isnull().T.any())}\n')
    prediction = pd.read_csv("D:\Work info\WestUnion\data\origin\\haolinju\\fresh-forecast-order.csv",
                             dtype={'bg_sort': str, 'md_sort': str, 'sm_sort': str, 'code': str},
                             names=['Unnamed', 'organ', 'class', 'bg_sort', 'bg_sort_name', 'md_sort', 'md_sort_name',
                                    'sm_sort', 'sm_sort_name', 'code', 'name', 'busdate', 'theory_sale', 'real_sale',
                                    'predict', 'advise_order', 'real_order'], header=0)
    print(f'\nprediction\n\nshape: {prediction.shape}\n\ndtypes:\n{prediction.dtypes}\n\nisnull-columns:\n'
          f'{prediction.isnull().any()}\n\nisnull-rows:\n{sum(prediction.isnull().T.any())}\n')
    promotion = pd.read_csv("D:\Work info\WestUnion\data\origin\haolinju\\promotion.csv", dtype={'code': str})
    print(f'\npromotion\n\nshape: {promotion.shape}\n\ndtypes:\n{promotion.dtypes}\n\nisnull-columns:\n'
          f'{promotion.isnull().any()}\n\nisnull-rows:\n{sum(promotion.isnull().T.any())}\n')

    comodt_meat = commodity[commodity['bg_sort_name'].str.contains('肉')]
    acct_meat = account[account['code'].isin(comodt_meat['code'])]
    run_meat = running[running['code'].isin(comodt_meat['code'])]
    stok_meat = stock[stock['code'].isin(comodt_meat['code'])]
    pred_meat = prediction[prediction['code'].isin(comodt_meat['code'])]
    prom_meat = promotion[promotion['code'].isin(comodt_meat['code'])]
    com_acct_meat = pd.merge(comodt_meat, acct_meat, how='left', on=['class', 'code'])
    com_acct_stk_meat = pd.merge(com_acct_meat, stok_meat, how='left', on=['organ', 'class', 'code', 'busdate'])
    com_acct_stk_meat.drop(columns=['name', 'sm_sort_name', 'md_sort_name', 'bg_sort_name'], inplace=True)
    pred_meat = pred_meat.drop(columns=['Unnamed', 'bg_sort_name', 'md_sort_name', 'sm_sort_name', 'name'])
    com_acct_stk_pred_meat = pd.merge(com_acct_stk_meat, pred_meat, how='outer',
                                      on=['organ', 'class', 'bg_sort', 'md_sort', 'sm_sort', 'code', 'busdate'])
    com_acct_stk_pred_meat.to_csv("D:\Work info\WestUnion\data\processed\haolinju\\com_acct_stk_pred_meat.csv",
                                  encoding='utf_8_sig', index=False)
    run_meat.to_csv("D:\Work info\WestUnion\data\processed\haolinju\\run_meat.csv", encoding='utf_8_sig', index=False)
