import pandas as pd
from prophet import Prophet


# 构造一个行数为1000的df，用于prophet的训练和预测
df = pd.DataFrame({
  'ds': pd.date_range('2021-01-01', periods=1000),
  'y': range(1000)
})
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
