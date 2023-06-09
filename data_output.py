import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 6)

input_path = r'D:\Work info\WestUnion\data\processed\HLJ\脱敏及筛选后样本数据\output'
output_path = r"D:\Work info\SCU\MathModeling\2023\data\output"

commodity = pd.read_csv(f'{input_path}/commodity.csv')

account = pd.read_csv(f'{input_path}/account.csv')
account.drop(columns=['amount', 'sum_price', 'sum_disc'], inplace=True)
account['busdate'] = pd.to_datetime(account['busdate'])

running = pd.read_csv(f'{input_path}/running.csv')
running.drop(columns='sum_sell', inplace=True)
running['busdate'] = pd.to_datetime(running['selltime'])

order = pd.read_csv(f'{input_path}/订货数据.csv')
order.drop(columns=['order_pred', 'loss_theory'], inplace=True)
order['busdate'] = pd.to_datetime(order['busdate'])
order.sort_values(by=['busdate', 'code'], inplace=True)

account = account[account['busdate'] >= order['busdate'].min()]
running = running[running['busdate'] >= order['busdate'].min()]

# 将account和order按['organ', 'class', 'code', 'busdate']合并
account_order = pd.merge(account, order, on=['organ', 'class', 'code', 'busdate'], how='left')

commodity.to_excel(f'{output_path}/commodity.xlsx', index=False)
running.to_csv(f'{output_path}/running.csv', index=False)
account_order.to_excel(f'{output_path}/account_order.xlsx', index=False)
