import os
import pandas as pd
from dateutil.parser import parse
from chinese_calendar import is_workday, is_holiday
import seaborn as sns
import matplotlib.pyplot as plt
from data_output import output_path_self_use

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)


weekday_dict = {
    1: "星期一",
    2: "星期二",
    3: "星期三",
    4: "星期四",
    5: "星期五",
    6: "星期六",
    7: "星期日",
}

input_path = output_path_self_use
script_name = os.path.basename(__file__).split(".")[0]
base_path = (
    "D:\\Work info\\SCU\\MathModeling\\2023\\data\\processed\\" + script_name + "\\"
)


code_sm = pd.read_excel(f"{input_path}" + "附件1-单品-分类.xlsx", sheet_name="Sheet1")
code_sm[["单品编码", "分类编码"]] = code_sm[["单品编码", "分类编码"]].astype(str)
code_sm.rename(
    columns={
        "单品编码": "code",
        "单品名称": "name",
        "分类编码": "sm_sort",
        "分类名称": "sm_sort_name",
    },
    inplace=True,
)
print(code_sm.dtypes, "\n")


sm_sort_name = code_sm["sm_sort_name"].unique()
output_path = [
    os.path.join(base_path, sm_sort_name[i] + "\\") for i in range(len(sm_sort_name))
]
for path in output_path:
    os.makedirs(path, exist_ok=True)


run_code = pd.read_excel(
    f"{input_path}" + "附件2-流水-销量-售价.xlsx", sheet_name="Sheet1"
)
run_code["单品编码"] = run_code["单品编码"].astype(str)
run_code_seg = run_code[run_code["销售类型"] == "销售"].copy()
run_code_seg.drop(
    columns=["销售单价(元/千克)", "销售类型", "是否打折销售"], inplace=True
)
run_code_seg.rename(
    columns={
        "销售日期": "busdate",
        "扫码销售时间": "selltime",
        "单品编码": "code",
        "销量(千克)": "amount",
    },
    inplace=True,
)
print(run_code_seg.dtypes, "\n")

run_merge = run_code_seg.merge(code_sm, how="left", on="code")

for i in range(len(sm_sort_name)):
    # 获取每一个小分类的流水df
    run_seg = run_merge[run_merge["sm_sort_name"] == sm_sort_name[i]]
    run_seg["selltime"] = pd.to_datetime(
        run_seg["selltime"], infer_datetime_format=True
    ).dt.strftime("%H:%M:%S")
    run_seg["selltime"] = run_seg["selltime"].apply(lambda x: parse(x).time())
    print(run_seg.dtypes, "\n")

    # 在run_seg中添加一列，用于标记星期
    run_seg["weekday"] = run_seg["busdate"].apply(
        lambda x: weekday_dict[x.weekday() + 1]
    )
    # 在run_seg中添加一列，根据busdate识别中国节假日
    # 创建一个新列，如果busdate是节假日，则值为True，否则为False
    run_seg["is_holiday"] = run_seg["busdate"].apply(lambda x: is_holiday(x))
    # 创建一个新列，如果busdate是工作日，则值为True，否则为False
    run_seg["is_workday"] = run_seg["busdate"].apply(lambda x: is_workday(x))
    # 判断is_holiday和is_workday是否是取反的关系
    assert (
        run_seg[
            (run_seg["is_holiday"] == True) & (run_seg["is_workday"] == True)
        ].shape[0]
    ) == 0, "is_holiday和is_workday不是取反的关系"
    # 去掉节假日的样本
    run_seg = run_seg[run_seg["is_holiday"] == False]
    run_seg_weekday = (
        run_seg.groupby(["weekday", "selltime"])["amount"].mean().reset_index()
    )

    # 将selltime转换为datetime类型，以便于后续的resample操作
    # 先转为字符串，才能使用pd.to_datetime()函数
    run_seg_weekday["selltime"] = run_seg_weekday["selltime"].astype(str)
    # 将 'selltime' 列转换为 datetime 类型
    run_seg_weekday["selltime"] = pd.to_datetime(run_seg_weekday["selltime"])

    # 以每隔15分钟为时间区间，进行分组求和，即每隔15分钟的销量之和转换为一个点的销量
    run_seg_weekday_resampled = (
        run_seg_weekday.groupby("weekday")
        .resample("15T", on="selltime")
        .sum(numeric_only=True)
        .reset_index()
    )
    # 将 'selltime' 列转换为 str 类型，且只保留时分
    run_seg_weekday_resampled["selltime"] = run_seg_weekday_resampled[
        "selltime"
    ].dt.strftime("%H:%M")
    # 按weekday列进行排序取值不同拆分成不同的dataframe
    run_seg_weekday_resampled_dict = dict(
        tuple(run_seg_weekday_resampled.groupby("weekday"))
    )

    # 将每个dataframe中的数据导出到excel
    for key, value in run_seg_weekday_resampled_dict.items():
        value.to_excel(f"{output_path[i]}" + f"{key}.xlsx", index=False)

    # 对 run_seg_weekday_resampled_dict 中的每一个df进行绘图
    for j, (key, df) in enumerate(run_seg_weekday_resampled_dict.items()):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="selltime", y="amount", data=df)
        plt.title(f"{sm_sort_name[i]}_{key}_日内平均销量曲线")
        plt.xlabel("日内销售时间段")
        plt.ylabel("平均销量")
        xticks = df["selltime"].unique()
        plt.xticks(xticks, rotation=90)
        plt.tight_layout()
        plt.savefig(
            f"{output_path[i]}" + f"{sm_sort_name[i]}_{key}_日内平均销量曲线.svg",
            format="svg",
        )
        plt.show()
