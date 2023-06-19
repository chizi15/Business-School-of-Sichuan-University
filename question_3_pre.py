import pandas as pd
import random


ratio = 0.9

# 生成teachers_use_data
running = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\running.csv")
run_apple = running[running['class'] == '水果课']
run_apple['busdate'] = pd.to_datetime(run_apple['selldate'], infer_datetime_format=True)
run_apple.sort_values(by=['busdate', 'selltime'], inplace=True)
run_apple.drop(columns=['selldate', 'sum_disc'], inplace=True)
run_apple.rename(columns={'sum_sell': 'sum_price'}, inplace=True)
# run_apple.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\teachers_use\data\run_apple_after_processed.xlsx", index=False, sheet_name='经过缺货填充的流水表_苹果')
run_apple.to_csv(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\teachers_use\data\run_apple_after_processed.csv", index=False)

account = pd.read_csv(r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata\account.csv")
account_apple = account[account['class'] == '水果课']
account_apple['busdate'] = pd.to_datetime(account_apple['busdate'], infer_datetime_format=True)
account_apple.drop(columns=['sum_disc'], inplace=True)
account_apple.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\teachers_use\data\account_apple_after_processed.xlsx", index=False, sheet_name='经过缺货填充的账表_苹果')


# 生成students_use_data
# 随机选取不做剔除的部分日期
dates_to_keep = set(running['selldate'].unique())
dates_to_keep = set(random.sample(dates_to_keep, k=int(len(dates_to_keep)*ratio)))
# Delete the samples after 20:00:00 in the dates that are not selected
run_apple_stds = run_apple[~((~run_apple['busdate'].isin(dates_to_keep)) & (run_apple['selltime'] > '20:00:00'))]
# run_apple_stds.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\students_use_data\run_apple_unprocessed.xlsx", index=False, sheet_name='未经过缺货填充的原始流水表_苹果')
run_apple_stds.to_csv(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\students_use_data\run_apple_unprocessed.csv", index=False)

run_apple_sum = run_apple.groupby(['busdate', 'code'])[['amount', 'sum_price']].sum().reset_index()
account_apple_run = run_apple_sum.groupby(['busdate'])[['amount', 'sum_price']].mean().reset_index()

account_seg = account_apple.groupby(['busdate'])[['sum_cost']].mean().reset_index()

account_apple_stds = pd.merge(account_apple_run, account_seg, on='busdate', how='left')
account_apple_stds.to_excel(r"D:\Work info\SCU\MathModeling\2023\data\processed\question_3\students_use_data\account_apple_unprocessed.xlsx", index=False, sheet_name='未经过缺货填充的账表_苹果')
