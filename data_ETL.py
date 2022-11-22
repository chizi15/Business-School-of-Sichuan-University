import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 20)


# read and summerize data
account = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\\account.csv")
account['busdate'] = pd.to_datetime(account['busdate'], infer_datetime_format=True)
account['code'] = account['code'].astype('str')
acct_grup = account.groupby(["organ", "code"])
print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n{account.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_grup)}\n')
pred = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\prediction.csv")
pred['busdate'] = pd.to_datetime(pred['busdate'])
pred['code'] = pred['code'].astype('str')
pred_grup = pred.groupby(["organ", "code"])
print(f'\npred\n\nshape: {pred.shape}\n\ndtypes:\n{pred.dtypes}\n\nisnull-columns:\n{pred.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(pred.isnull().T.any())}\n\nnumber of commodities:\n{len(pred_grup)}\n')
commodity = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\commodity.csv")
commodity[['class', 'sort']] = commodity[['class', 'sort']].astype('str')
comodt_grup = commodity.groupby(['code'])
print(f'\ncommodity\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n{commodity.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(commodity.isnull().T.any())}\n\nnumber of commodities:\n{len(comodt_grup)}\n')
stock = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\stock.csv")
stock['busdate'] = pd.to_datetime(stock['busdate'])
stock['code'] = stock['code'].astype('str')
stock_grup = stock.groupby(["organ", "code"])
print(f'\nstock\n\nshape: {stock.shape}\n\ndtypes:\n{stock.dtypes}\n\nisnull-columns:\n{stock.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(stock.isnull().T.any())}\n\nnumber of commodities:\n{len(stock_grup)}\n')
running = pd.read_csv("D:\Work info\WestUnion\data\origin\dehui\\running.csv",
                      parse_dates=['selldate'], dtype={'code': str})
running['selltime'] = running['selltime'].apply(lambda x: x[:8])  # 截取出时分秒
running['selltime'] = pd.to_datetime(running['selltime'], format='%H:%M:%S')
running['selltime'] = running['selltime'].dt.time  # 去掉to_datetime自动生成的年月日
run_grup = running.groupby(["organ", "code"])
print(f'\nrunning\n\nshape: {running.shape}\n\ndtypes:\n{running.dtypes}\n\nisnull-columns:\n{running.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(running.isnull().T.any())}\n\nnumber of commodities:\n{len(run_grup)}\n')

# merge data
acct_pred = pd.merge(account, pred, how='left', on=['organ', 'code', 'busdate'])
acct_pred_grup = acct_pred.groupby(["organ", "code"])
print(f'\nacct_pred\n\nshape: {acct_pred.shape}\n\ndtypes:\n{acct_pred.dtypes}\n\nisnull-columns:\n{acct_pred.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(acct_pred.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_pred_grup)}\n')
acct_pred_com = pd.merge(acct_pred, commodity, how='inner', on=['code'])
acct_pred_com_grup = acct_pred_com.groupby(["organ", "code"])
print(f'\nacct_pred_com\n\nshape: {acct_pred_com.shape}\n\ndtypes:\n{acct_pred_com.dtypes}\n\nisnull-columns:\n{acct_pred_com.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(acct_pred_com.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_pred_com_grup)}\n')
acct_pred_com_stk = pd.merge(acct_pred_com, stock, how='left', on=['organ', 'code', 'busdate'])
acct_pred_com_stk_grup = acct_pred_com_stk.groupby(["organ", "code"])
print(f'\nacct_pred_com_stk\n\nshape: {acct_pred_com_stk.shape}\n\ndtypes:\n{acct_pred_com_stk.dtypes}\n\nisnull-columns:\n{acct_pred_com_stk.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(acct_pred_com_stk.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_pred_com_stk_grup)}\n')
running_grup = running.groupby(['organ', 'code', 'selldate'], as_index=False).sum()
running_grup.rename(columns={'selldate': 'busdate', 'amount': 'amount_run', 'sum_sell': 'sum_price_run'}, inplace=True)
acct_pred_com_stk_run = pd.merge(acct_pred_com_stk, running_grup, how='left', on=['organ', 'code', 'busdate'])

# export to sheet
chosen = 1
if chosen == 1:
    chosen = 'origin'
elif chosen == 2:
    chosen = 'drop_fcst_na'
elif chosen == 3:
    chosen = 'exact'
else:
    chosen = 'exact_drop'
match chosen:
    case 'origin':
        acct_pred_com_stk_run.to_csv(f"D:\Work info\WestUnion\data\processed\dehui\\acct_pred_com_stk_run_{chosen}.csv",
                                     encoding='utf_8_sig', index=False)
    case 'drop_fcst_na':
        acct_pred_com_stk_run.dropna(subset=['fcstamou', 'amount'], inplace=True)
        acct_pred_com_stk_run.to_csv(f"D:\Work info\WestUnion\data\processed\dehui\\acct_pred_com_stk_run_{chosen}.csv",
                                     encoding='utf_8_sig', index=False)
    case 'exact':
        acct_pred_com_stk_run = acct_pred_com_stk_run[(acct_pred_com_stk_run['amount_run'] == acct_pred_com_stk_run['amount']) &
            (acct_pred_com_stk_run['sum_price_run'] == acct_pred_com_stk_run['sum_price'])]
        acct_pred_com_stk_run.to_csv(f"D:\Work info\WestUnion\data\processed\dehui\\acct_pred_com_stk_run_{chosen}.csv",
                                     encoding='utf_8_sig', index=False)
    case _:
        acct_pred_com_stk_run = acct_pred_com_stk_run[(acct_pred_com_stk_run['amount_run'] == acct_pred_com_stk_run['amount']) &
                                              (acct_pred_com_stk_run['sum_price_run'] == acct_pred_com_stk_run[
                                                  'sum_price'])]
        acct_pred_com_stk_run.dropna(subset=['fcstamou', 'amount'], inplace=True)
        acct_pred_com_stk_run.to_csv(f"D:\Work info\WestUnion\data\processed\dehui\\acct_pred_com_stk_run_{chosen}.csv",
                                     encoding='utf_8_sig', index=False)

acct_pred_com_stk_run_grup = acct_pred_com_stk_run.groupby(["organ", "code"])
print(f'\nacct_pred_com_stk_run\n\nshape: {acct_pred_com_stk_run.shape}\n\ndtypes:\n{acct_pred_com_stk_run.dtypes}'
      f'\n\nisnull-columns:\n{acct_pred_com_stk_run.isnull().any()}\n\nisnull-rows:\n'
      f'{sum(acct_pred_com_stk_run.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_pred_com_stk_run_grup)}\n')
