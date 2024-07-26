# -*- coding: utf-8 -*-
from prophet import Prophet

# from gluonts.ext.prophet import ProphetPredictor
# from pycaret.time_series import TSForecastingExperiment
import pandas as pd
import numpy as np
from scipy import stats
import chinese_calendar
import fitter
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)  # regression_evaluation_main所在文件夹的绝对路径
from regression_evaluation_main import regression_evaluation_def as ref
from data_output import output_path_self_use, last_day

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
"""
中文字体	    说明
‘SimHei’	中文黑体
‘Kaiti’	    中文楷体
‘LiSu’	    中文隶书
‘FangSong’	中文仿宋
‘YouYuan’	中文幼圆
’STSong‘    华文宋体
"""
print("Imported packages successfully.", "\n")


# 设置全局参数
periods = 7  # 预测步数
output_index = 7  # 将前output_index个预测结果作为最终结果
extend_power = 1 / 5  # 数据扩增的幂次
interval_width = 0.95  # prophet的置信区间宽度
last_day = last_day  # 训练集的最后一天+1，即预测集的第一天
mcmc_samples = 0  # mcmc采样次数
days = 30  # 用于计算平均销量的距最后一天的天数
distributions = [
    "cauchy",
    "chi2",
    "expon",
    "exponpow",
    "gamma",
    "lognorm",
    "norm",
    "powerlaw",
    "irayleigh",
    "uniform",
]
seasonality_mode = "multiplicative"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", periods)

input_path = output_path_self_use
# create the directory if it doesn't exist
output_path = r"D:\Work info\SCU\MathModeling\2023\data\processed\question_2"
if not os.path.exists(output_path):
    os.makedirs(output_path)


# 数据预处理
code_sm = pd.read_excel(f"{input_path}" + "附件1-单品-分类.xlsx", sheet_name="Sheet1")
code_sm[["单品编码", "分类编码"]] = code_sm[["单品编码", "分类编码"]].astype(str)
print(
    f"code_sm['单品编码'].nunique(): {code_sm['单品编码'].nunique()}\ncode_sm['单品名称'].nunique(): {code_sm['单品名称'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\n"
)

run_code = pd.read_excel(
    f"{input_path}" + "附件2-流水-销量-售价.xlsx", sheet_name="Sheet1"
)
run_code["单品编码"] = run_code["单品编码"].astype(str)
print(f"run_code['单品编码'].nunique(): {run_code['单品编码'].nunique()}\n")

cost_code = pd.read_excel(f"{input_path}" + "附件3-进价-单品.xlsx", sheet_name="Sheet1")
loss_code = pd.read_excel(f"{input_path}" + "附件4-损耗-单品.xlsx", sheet_name="Sheet1")


code_sm.rename(
    columns={
        "单品编码": "code",
        "单品名称": "name",
        "分类编码": "sm_sort",
        "分类名称": "sm_sort_name",
    },
    inplace=True,
)
run_code.drop(columns=["扫码销售时间", "销售类型", "是否打折销售"], inplace=True)
run_code.rename(
    columns={
        "单品编码": "code",
        "销售日期": "busdate",
        "销量(千克)": "amount",
        "销售单价(元/千克)": "price",
    },
    inplace=True,
)
cost_code.rename(
    columns={"单品编码": "code", "日期": "busdate", "批发价格(元/千克)": "unit_cost"},
    inplace=True,
)
loss_code.rename(
    columns={"单品编码": "code", "单品名称": "name", "损耗率(%)": "loss_rate"},
    inplace=True,
)
cost_code = cost_code.astype({"code": "str"})
loss_code = loss_code.astype({"code": "str"})


run_code["sum_price"] = run_code["amount"] * run_code["price"]
acct_code = (
    run_code.groupby(["code", "busdate"])[["amount", "sum_price"]].sum().reset_index()
)
acct_code["price"] = acct_code["sum_price"] / acct_code["amount"]

acct_code_sm = acct_code.merge(code_sm, on="code", how="left")
acct_code_sm_loss = acct_code_sm.merge(loss_code, on=["code", "name"], how="left")
acct_code_sm_loss_cost = acct_code_sm_loss.merge(
    cost_code, on=["code", "busdate"], how="left"
)
acct_code_sm_loss_cost["sum_cost"] = (
    acct_code_sm_loss_cost["amount"] * acct_code_sm_loss_cost["unit_cost"]
)
acct_code_sm_loss_cost.drop(columns=["price"], inplace=True)
acct_sm = (
    acct_code_sm_loss_cost.groupby(["sm_sort", "sm_sort_name", "busdate"])[
        ["amount", "sum_cost", "sum_price"]
    ]
    .sum()
    .reset_index()
)
acct_loss = (
    acct_code_sm_loss_cost.groupby(["sm_sort", "sm_sort_name", "busdate"])[
        ["loss_rate"]
    ]
    .mean()
    .reset_index()
)
acct_sm_loss = acct_sm.merge(
    acct_loss, on=["sm_sort", "sm_sort_name", "busdate"], how="left"
)
acct_sm_loss["unit_cost"] = acct_sm_loss["sum_cost"] / acct_sm_loss["amount"]
print(
    f"\nacct_sm_loss\n\nshape: {acct_sm_loss.shape}\n\ndtypes:\n{acct_sm_loss.dtypes}\n\nnull-columns:\n{acct_sm_loss.isnull().sum()}",
    "\n",
)
acct_sm_loss = acct_sm_loss.round(3)
acct_sm_loss.sort_values(by=["sm_sort", "busdate"], inplace=True)

# 考虑损耗率
acct_sm_loss["amount_origin"] = acct_sm_loss["amount"]
acct_sm_loss["amount"] = acct_sm_loss["amount"] / (1 - acct_sm_loss["loss_rate"] / 100)
acct_sm_loss["sum_cost"] = acct_sm_loss["amount"] * acct_sm_loss["unit_cost"]
acct_sm_loss["sum_price"] = acct_sm_loss["sum_price"] / (
    1 - acct_sm_loss["loss_rate"] / 100
)
acct_sm_loss["profit"] = (
    acct_sm_loss["sum_price"] - acct_sm_loss["sum_cost"]
) / acct_sm_loss["sum_price"]
acct_sm_loss = acct_sm_loss[acct_sm_loss["profit"] > 0]
if (
    abs(
        np.mean(
            acct_sm_loss["sum_cost"] / acct_sm_loss["amount"]
            - acct_sm_loss["unit_cost"]
        )
    )
    < 1e-3
):
    print("The mean of (sum_cost / amount - unit_cost) is less than 1e-3")
else:
    raise ValueError(
        "The mean of (sum_cost / amount - unit_cost) is not less than 1e-3"
    )


# question_2；对sm_sort层级进行预测和定价，而不是在code层级上进行预测和定价
for i in acct_sm_loss["sm_sort_name"].unique():
    acct_ori = acct_sm_loss[acct_sm_loss["sm_sort_name"] == i]
    acct_ori["price"] = acct_ori["sum_price"] / acct_ori["amount"]
    sm_qielei_all = acct_sm_loss[acct_sm_loss["sm_sort_name"] == i]

    # ts = TSForecastingExperiment()
    # ts.setup(data=sm_qielei_all, target='amount')
    # Prophet = ts.create_model(model='prophet', seasonality_mode=seasonality_mode, mcmc_samples=mcmc_samples, interval_width=interval_width)

    amount_base = np.mean(
        [
            sm_qielei_all["amount"][-days:].mean(),
            np.percentile(sm_qielei_all["amount"][-days:], 50),
        ]
    )
    amount_origin_base = np.mean(
        [
            sm_qielei_all["amount_origin"][-days:].mean(),
            np.percentile(sm_qielei_all["amount_origin"][-days:], 50),
        ]
    )

    # 用sm_qielei_all的最后一行复制出periods行，并将busdate分别加上1、2、3、4、5、6、7，得到预测期的日期
    sm_qielei_all = sm_qielei_all.append(
        [sm_qielei_all.iloc[-1, :]] * (periods), ignore_index=True
    )
    sm_qielei_all["busdate"][-(periods):] = sm_qielei_all["busdate"][
        -(periods):
    ] + pd.to_timedelta(np.arange(1, periods + 1), unit="d")

    sm_qielei_all.loc[sm_qielei_all.index[-periods:], "amount"] = (
        amount_base
        + np.random.normal(
            0, sm_qielei_all["amount"].std() / sm_qielei_all["amount"].mean(), periods
        )
    )  # 为预测期的销量添加随机噪声，防止后面metrics计算中MGC报错
    sm_qielei_all.loc[sm_qielei_all.index[-periods:], "amount_origin"] = (
        amount_origin_base
        + np.random.normal(
            0,
            sm_qielei_all["amount_origin"].std()
            / sm_qielei_all["amount_origin"].mean(),
            periods,
        )
    )

    # 将sm_qielei_all中的amount_origin列的值赋给amount列，如果amount列的值小于amount_origin列的值
    sm_qielei_all[-(periods):].loc[
        sm_qielei_all[-(periods):]["amount"]
        < sm_qielei_all[-(periods):]["amount_origin"],
        "amount",
    ] = sm_qielei_all[-(periods):]["amount_origin"] + np.random.random(periods)

    # 将全集上busdate缺失不连续的行用插值法进行填充
    sm_qielei_all_col = (
        sm_qielei_all.set_index("busdate")
        .resample("D")
        .mean()
        .interpolate(method="linear")
        .reset_index()
    )
    print(
        "新增插入的行是：",
        "\n",
        pd.merge(
            sm_qielei_all_col, sm_qielei_all, on="busdate", how="left", indicator=True
        ).query('_merge == "left_only"'),
        "\n",
    )
    sm_qielei_all = pd.merge(sm_qielei_all_col, sm_qielei_all, on="busdate", how="left")
    sm_qielei_all.drop(
        columns=[
            "amount_y",
            "sum_cost_y",
            "sum_price_y",
            "unit_cost_y",
            "loss_rate_y",
            "amount_origin_y",
            "profit_y",
        ],
        inplace=True,
    )
    sm_qielei_all.rename(
        columns={
            "amount_x": "amount",
            "sum_cost_x": "sum_cost",
            "sum_price_x": "sum_price",
            "unit_cost_x": "unit_cost",
            "loss_rate_x": "loss_rate",
            "profit_x": "profit",
            "amount_origin_x": "amount_origin",
        },
        inplace=True,
    )

    print(
        f"{i}经过差值后存在空值的字段有:\n{sm_qielei_all.columns[sm_qielei_all.isnull().sum()>0].to_list()}\n"
    )
    pd.set_option("display.max_rows", 20)
    print(
        f"存在空值的字段的数据类型为：\n{sm_qielei_all[sm_qielei_all.columns[sm_qielei_all.isnull().sum()>0]].dtypes}\n"
    )
    pd.set_option("display.max_rows", 7)
    print(
        f"{i}经过差值后存在空值的行数为：{sm_qielei_all.isnull().T.any().sum()}", "\n"
    )
    # 将sm_qielei_all中存在空值的那些字段的空值用众数填充
    for col in sm_qielei_all.columns[sm_qielei_all.isnull().sum() > 0]:
        sm_qielei_all[col].fillna(sm_qielei_all[col].mode()[0], inplace=True)
    print(
        f"{i}经过众数填充后存在空值的行数为：{sm_qielei_all.isnull().T.any().sum()}",
        "\n",
    )

    sm_qielei_all = sm_qielei_all.round(3)
    sm_qielei_all.to_excel(
        os.path.join(output_path, f"{i}_在全集上的样本.xlsx"),
        index=False,
        sheet_name=f"{i}在全集上的样本",
    )
    # 获取茄类在训练集上的样本
    sm_qielei = sm_qielei_all[:-periods]
    # 将训练集上非正毛利的异常样本的sum_price重新赋值
    sm_qielei.loc[sm_qielei["sum_cost"] >= sm_qielei["sum_price"], "sum_price"] = (
        sm_qielei.loc[sm_qielei["sum_cost"] >= sm_qielei["sum_price"], "sum_cost"]
        * max(
            sm_qielei_all["sum_price"].mean() / sm_qielei_all["sum_cost"].mean(), 1.01
        )
    )
    if sum(sm_qielei["sum_cost"] >= sm_qielei["sum_price"]) > 0:
        raise ValueError("There are still some negative profit in sm_qielei")

    sm_qielei = sm_qielei.round(3)
    sm_qielei.to_excel(
        output_path + f"\{i}_在训练集上的样本.xlsx",
        index=False,
        sheet_name=f"{i}在训练集上的样本",
    )

    # 对sm_qielei_all中数值型变量的列：amount，sum_cost和sum_price画时序图
    sm_qielei_all_num = sm_qielei_all.select_dtypes(include=np.number)
    for col in sm_qielei_all_num:
        sns.lineplot(x="busdate", y=col, data=sm_qielei_all)
        plt.xticks(rotation=45)
        plt.xlabel("busdate")
        plt.ylabel(col)
        plt.title(f"{i}Time Series Graph")
        plt.show()

    # 输出这三条时序图中，非空数据的起止日期，用循环实现
    for col in sm_qielei_all_num:
        print(
            f'{col}非空数据的起止日期为：{sm_qielei_all[sm_qielei_all[col].notnull()]["busdate"].min()}到{sm_qielei_all[sm_qielei_all[col].notnull()]["busdate"].max()}',
            "\n",
        )

    # 断言这三个字段非空数据的起止日期相同
    assert (
        sm_qielei_all[sm_qielei_all["amount"].notnull()]["busdate"].min()
        == sm_qielei_all[sm_qielei_all["sum_cost"].notnull()]["busdate"].min()
        == sm_qielei_all[sm_qielei_all["sum_price"].notnull()]["busdate"].min()
    ), "三个字段非空数据的开始日期不相同"
    assert (
        sm_qielei_all[sm_qielei_all["amount"].notnull()]["busdate"].max()
        == sm_qielei_all[sm_qielei_all["sum_cost"].notnull()]["busdate"].max()
        == sm_qielei_all[sm_qielei_all["sum_price"].notnull()]["busdate"].max()
    ), "三个字段非空数据的结束日期不相同"

    # 用prophet获取训练集上的星期效应系数、节日效应系数和年季节性效应系数
    qielei_prophet_amount = sm_qielei[["busdate", "amount"]].rename(
        columns={"busdate": "ds", "amount": "y"}
    )
    m_amount = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode="multiplicative",
        holidays_prior_scale=10,
        seasonality_prior_scale=10,
        mcmc_samples=mcmc_samples,
        interval_width=interval_width,
    )
    m_amount.add_country_holidays(country_name="CN")
    m_amount.fit(qielei_prophet_amount)
    future_amount = m_amount.make_future_dataframe(periods=periods)
    forecast_amount = m_amount.predict(future_amount)
    fig1 = m_amount.plot(forecast_amount)
    plt.show()
    fig2 = m_amount.plot_components(forecast_amount)
    plt.show()
    fig1.savefig(output_path + f"\{i}_fit_amount.svg", dpi=300, bbox_inches="tight")
    fig2.savefig(
        output_path + f"\{i}_fit_amount_components.svg", dpi=300, bbox_inches="tight"
    )

    holiday_effect = forecast_amount[
        ["ds", "holidays", "holidays_lower", "holidays_upper"]
    ]
    weekly_effect = forecast_amount[["ds", "weekly", "weekly_lower", "weekly_upper"]]
    yearly_effect = forecast_amount[["ds", "yearly", "yearly_lower", "yearly_upper"]]
    multiplicative_terms = forecast_amount[
        [
            "ds",
            "multiplicative_terms",
            "multiplicative_terms_lower",
            "multiplicative_terms_upper",
        ]
    ]

    # 根据qielei_prophet的ds列，将holiday_effect, weekly_effect, yearly_effect, multiplicative_terms左连接合并到qielei_prophet中
    qielei_prophet_amount = pd.merge(
        qielei_prophet_amount, holiday_effect, on="ds", how="left"
    )
    qielei_prophet_amount = pd.merge(
        qielei_prophet_amount, weekly_effect, on="ds", how="left"
    )
    qielei_prophet_amount = pd.merge(
        qielei_prophet_amount, yearly_effect, on="ds", how="left"
    )
    qielei_prophet_amount = pd.merge(
        qielei_prophet_amount, multiplicative_terms, on="ds", how="left"
    )
    qielei_prophet_amount = qielei_prophet_amount.rename(
        columns={
            "holidays": "holiday_effect",
            "weekly": "weekly_effect",
            "yearly": "yearly_effect",
            "multiplicative_terms": "total_effect",
        }
    )
    print(qielei_prophet_amount.isnull().sum(), "\n")

    # 在qielei_prophet中，增加中国日历的星期和节假日的列，列名分别为weekday和holiday
    qielei_prophet_amount["weekday"] = qielei_prophet_amount[
        "ds"
    ].dt.weekday  # 0-6, Monday is 0
    # 根据日期获取中国节日名称，使用chinese_calendar库
    qielei_prophet_amount["holiday"] = qielei_prophet_amount["ds"].apply(
        lambda x: (
            chinese_calendar.get_holiday_detail(x)[1]
            if chinese_calendar.get_holiday_detail(x)[0]
            else None
        )
    )
    pd.set_option("display.max_rows", 20)
    print("空值填充前，各列空值数", "\n", qielei_prophet_amount.isnull().sum(), "\n")
    pd.set_option("display.max_rows", 7)
    # 若qielei_prophet_amount中存在空值，则填充为该列前7天的均值
    for col in qielei_prophet_amount.columns[1:-2]:
        print(col)
        if qielei_prophet_amount[col].isnull().sum() > 0:
            qielei_prophet_amount[col] = qielei_prophet_amount[col].fillna(
                qielei_prophet_amount[col].rolling(7, min_periods=1).mean()
            )
    print()
    pd.set_option("display.max_rows", 20)
    print("空值填充后，各列空值数", "\n", qielei_prophet_amount.isnull().sum(), "\n")
    pd.set_option("display.max_rows", 7)
    # 保存输出带有时间效应和星期、节假日标签的茄类销量样本
    if seasonality_mode == "multiplicative":
        # 将qielei_prophet_amount中holiday_effect','weekly_effect','yearly_effect','total_effect'四列的值都限定到某个区间
        qielei_prophet_amount[
            ["holiday_effect", "weekly_effect", "yearly_effect", "total_effect"]
        ] = qielei_prophet_amount[
            ["holiday_effect", "weekly_effect", "yearly_effect", "total_effect"]
        ].apply(
            lambda x: np.clip(x, -0.9, 9)
        )
    else:
        qielei_prophet_amount[
            ["holiday_effect", "weekly_effect", "yearly_effect", "total_effect"]
        ] = qielei_prophet_amount[
            ["holiday_effect", "weekly_effect", "yearly_effect", "total_effect"]
        ].apply(
            lambda x: np.clip(x, np.percentile(x, 10), np.percentile(x, 90))
        )
    qielei_prophet_amount = qielei_prophet_amount.round(3)
    qielei_prophet_amount.to_excel(
        os.path.join(output_path, f"{i}_销量时序分解.xlsx"),
        index=False,
        sheet_name=f"{i}用历史销量计算出的时间效应，合并到训练集中",
    )

    # 验证prophet分解出的各个分项的计算公式
    print(
        f"在乘法模式下，trend*(1+multiplicative_terms)=yhat, 即：sum(forecast['trend']*(1+forecast['multiplicative_terms'])-forecast['yhat']) = {sum(forecast_amount['trend']*(1+forecast_amount['multiplicative_terms'])-forecast_amount['yhat'])}",
        "\n",
    )
    print(
        f"sum(forecast['multiplicative_terms']-(forecast['holidays']+forecast['weekly']+forecast['yearly'])) = {sum(forecast_amount['multiplicative_terms']-(forecast_amount['holidays']+forecast_amount['weekly']+forecast_amount['yearly']))}",
        "\n",
    )

    # 剔除sm_qielei的时间相关效应，得到sm_qielei_no_effect；因为multiplicative_terms包括节假日效应、星期效应、年季节性效应，multiplicative_terms就代表综合时间效应。
    if seasonality_mode == "multiplicative":
        sm_qielei["amt_no_effect"] = (
            sm_qielei["amount"].values
            / (1 + qielei_prophet_amount["total_effect"]).values
        )
    else:
        sm_qielei["amt_no_effect"] = (
            sm_qielei["amount"].values - qielei_prophet_amount["total_effect"].values
        )
    # 对sm_qielei['amt_no_effect']和sm_qielei['amount']画图，比较两者的差异，横坐标为busdate, 纵坐标为amount和amt_no_effect, 用plt画图
    fig = plt.figure()
    plt.plot(sm_qielei["busdate"], sm_qielei["amount"], label="剔除时间效应前销量")
    plt.plot(
        sm_qielei["busdate"], sm_qielei["amt_no_effect"], label="剔除时间效应后销量"
    )
    plt.xticks(rotation=45)
    plt.xlabel("销售日期")
    plt.ylabel(f"{i}销量")
    plt.title(f"{i}剔除时间效应前后，销量时序对比")
    plt.legend(loc="best")
    plt.show()
    fig.savefig(
        output_path + f"\{i}_剔除时间效应前后，{i}销量时序对比.svg",
        dpi=300,
        bbox_inches="tight",
    )

    # 计算sm_qielei['amt_no_effect']和sm_qielei['amount']的统计信息
    sm_qielei_amount_effect_compare = sm_qielei[["amount", "amt_no_effect"]].describe()
    print(sm_qielei_amount_effect_compare.round(3), "\n")
    sm_qielei_amount_effect_compare = sm_qielei_amount_effect_compare.round(3)
    sm_qielei_amount_effect_compare.to_excel(
        output_path + f"\{i}_剔除时间效应前后，历史销量的描述性统计信息对比.xlsx",
        sheet_name=f"{i}剔除时间效应前后，历史销量的描述性统计信息对比",
    )
    # 计算sm_qielei['amt_no_effect']和sm_qielei['amount']的相关系数
    print(sm_qielei[["amount", "amt_no_effect"]].corr().round(4), "\n")

    # 对历史销量数据进行扩增，使更近的样本占更大的权重
    sm_qielei_amt_ext = ref.extendSample(
        sm_qielei["amt_no_effect"].values,
        max_weight=int(len(sm_qielei["amt_no_effect"].values) ** (extend_power)),
    )

    # 在同一个图中，画出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的分布及概率密度函数的对比图
    fig, ax = plt.subplots(1, 1)
    sns.distplot(sm_qielei_amt_ext, ax=ax, label="extended")
    sns.distplot(sm_qielei["amt_no_effect"].values, ax=ax, label="original")
    ax.legend()
    plt.title(f"{i}数据扩增前后，历史销量的概率密度函数对比图")
    plt.show()
    fig.savefig(
        output_path + f"\{i}_数据扩增前后，历史销量的概率密度函数对比图.svg",
        dpi=300,
        bbox_inches="tight",
    )

    # 给出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的描述性统计
    sm_qielei_amt_ext_describe = pd.Series(
        sm_qielei_amt_ext, name=f"sm_{i}_amt_ext_describe"
    ).describe()
    sm_qielei_amt_describe = sm_qielei["amt_no_effect"].describe()
    sm_qielei_amt_ext_compare = pd.concat(
        [sm_qielei_amt_describe, sm_qielei_amt_ext_describe], axis=1
    ).rename(columns={"amt_no_effect": f"sm_{i}_amt_describe"})
    print(sm_qielei_amt_ext_compare.round(2), "\n")
    sm_qielei_amt_ext_compare = sm_qielei_amt_ext_compare.round(2)
    sm_qielei_amt_ext_compare.to_excel(
        output_path + f"\{i}_数据扩增前后，历史销量的描述性统计信息对比.xlsx",
        sheet_name=f"{i}数据扩增前后，历史销量的描述性统计信息对比",
    )

    # 给出sm_qielei_amt_ext和sm_qielei['amt_no_effect'].values的Shapiro-Wilk检验结果
    stat, p = stats.shapiro(sm_qielei_amt_ext)
    print('"sm_qielei_amt_ext" Shapiro-Wilk test statistic:', round(stat, 4))
    print('"sm_qielei_amt_ext" Shapiro-Wilk test p-value:', round(p, 4))
    # Perform Shapiro-Wilk test on amt_no_effect
    stat, p = stats.shapiro(sm_qielei["amt_no_effect"].values)
    print('"amt_no_effect" Shapiro-Wilk test statistic:', round(stat, 4))
    print('"amt_no_effect" Shapiro-Wilk test p-value:', round(p, 4), "\n")

    # 计算sm_qielei在训练集上的平均毛利率
    sm_qielei["price_daily_avg"] = sm_qielei["sum_price"] / sm_qielei["amount"]
    sm_qielei["cost_daily_avg"] = sm_qielei["sum_cost"] / sm_qielei["amount"]
    sm_qielei["profit_daily"] = (
        sm_qielei["price_daily_avg"] - sm_qielei["cost_daily_avg"]
    ) / sm_qielei["price_daily_avg"]
    profit_avg = max(
        (
            sm_qielei["profit_daily"].mean()
            + np.percentile(sm_qielei["profit_daily"], 50)
        )
        / 2,
        0.01,
    )
    # 用newsvendor模型计算sm_qielei_amt_ext的平稳订货量q_steady
    f = fitter.Fitter(sm_qielei_amt_ext, distributions="gamma")
    f.fit()
    q_steady = stats.gamma.ppf(profit_avg, *f.fitted_param["gamma"])
    params = f.fitted_param["gamma"]
    params_rounded = tuple(round(float(param), 4) for param in params)
    print(f"拟合分布的最优参数是: \n {params_rounded}", "\n")
    print(f"第一次的平稳订货量q_steady = {round(q_steady, 3)}", "\n")

    # 观察sm_qielei_amt_ext的分布情况
    f = fitter.Fitter(sm_qielei_amt_ext, distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f"\n{comparison_of_distributions_qielei.round(4)}\n")
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    comparison_of_distributions_qielei.to_excel(
        output_path + f"\{i}_comparison_of_distributions.xlsx",
        sheet_name=f"{i}_comparison of distributions",
    )

    name = list(f.get_best().keys())[0]
    print(f"best distribution: {name}" "\n")
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f"用于拟合分布的，{i}数据扩增后的历史销量")
    plt.ylabel("Probability")
    plt.title(f"{i} comparison of distributions")
    plt.show()
    figure.savefig(output_path + f"\{i}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, "b-.", label="f.y")
    plt.plot(f.x, f.fitted_pdf[name], "r-", label="f.fitted_pdf")
    plt.xlabel(f"用于拟合分布的，{i}数据扩增后的历史销量")
    plt.ylabel("Probability")
    plt.title(f"best distribution: {name}")
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"\{i}_best distribution.svg")
    figure.clear()

    # 将时间效应加载到q_steady上，得到预测期的最优订货量q_star
    train_set = qielei_prophet_amount[
        [
            "ds",
            "y",
            "holiday_effect",
            "weekly_effect",
            "yearly_effect",
            "holiday",
            "weekday",
        ]
    ]
    all_set = pd.merge(future_amount, train_set, on="ds", how="left")
    all_set["weekday"][-periods:] = all_set["ds"][
        -periods:
    ].dt.weekday  # 0-6, Monday is 0
    all_set["holiday"][-periods:] = all_set["ds"][-periods:].apply(
        lambda x: (
            chinese_calendar.get_holiday_detail(x)[1]
            if chinese_calendar.get_holiday_detail(x)[0]
            else None
        )
    )
    # 计算all_set[:-periods]中，weekly_effect字段关于weekday字段每个取值的平均数，并保留ds字段
    weekly_effect_avg = all_set[:-periods].groupby(["weekday"])["weekly_effect"].mean()
    # 将all_set和weekly_effect_avg按weekday字段进行左连接，只保留一个weekday字段
    all_set = pd.merge(
        all_set, weekly_effect_avg, on="weekday", how="left", suffixes=("", "_avg")
    ).drop(columns=["weekly_effect"])
    # 对预测期的节假日系数赋值
    if len(set(all_set["holiday"][-periods:])) == 1:
        all_set["holiday_effect"][-periods:] = 0
    else:
        raise ValueError("预测期中，存在多个节假日，需要手动设置holiday_effect")
    # 提取all_set['ds']中的年、月、日
    all_set["year"] = all_set["ds"].dt.year
    all_set["month"] = all_set["ds"].dt.month
    all_set["day"] = all_set["ds"].dt.day
    # 取all_set[-periods:]和all_set[:-periods]中，月、日相同，但年不同的样本，计算年效应
    yearly_effect_avg = (
        all_set[:-periods]
        .groupby(["month", "day"])["yearly_effect"]
        .mean()
        .reset_index()
    )
    all_set = pd.merge(
        all_set,
        yearly_effect_avg,
        on=["month", "day"],
        how="left",
        suffixes=("", "_avg"),
    ).drop(columns=["yearly_effect"])
    # 计算q_star
    q_star = q_steady * (
        1
        + (
            all_set["holiday_effect"]
            + all_set["weekly_effect_avg"]
            + all_set["yearly_effect_avg"]
        )[-periods:]
    )
    print("q_star = ", "\n", f"{round(q_star, 3)}", "\n")
    all_set["y"][-periods:] = q_star
    all_set["y"][:-periods] = forecast_amount["yhat"][:-periods]
    all_set.drop(columns=["year", "month", "day"], inplace=True)
    all_set.rename(columns={"y": "预测销量", "ds": "销售日期"}, inplace=True)
    all_set["训练集平均毛利率"] = profit_avg
    all_set["第一次计算的平稳订货量"] = q_steady
    all_set = all_set.round(3)
    all_set.to_excel(
        output_path + f"\{i}_全集上的第一次决策结果及时间效应系数.xlsx",
        index=False,
        encoding="utf-8-sig",
        sheet_name=f"问题2最终结果：{i}全集上的预测订货量及时间效应系数",
    )

    # 使用ref评估最后一周，即预测期的指标
    sm_qielei_all = sm_qielei_all[["busdate", "amount", "amount_origin"]].rename(
        columns={
            "busdate": "销售日期",
            "amount": "实际销量",
            "amount_origin": "原始销量",
        }
    )
    sm_qielei_all = pd.merge(sm_qielei_all, all_set, on="销售日期", how="left")
    sm_qielei_seg = sm_qielei_all[
        sm_qielei_all["销售日期"] >= str(sm_qielei_all["销售日期"].max().year)
    ]

    res = ref.regression_evaluation_single(
        y_true=sm_qielei_seg["实际销量"][-periods:].values,
        y_pred=sm_qielei_seg["预测销量"][-periods:].values,
    )
    accu_sin = ref.accuracy_single(
        y_true=sm_qielei_seg["实际销量"][-periods:].values,
        y_pred=sm_qielei_seg["预测销量"][-periods:].values,
    )
    metrics_values = [accu_sin] + list(res[:-2])
    metrics_names = [
        "AA",
        "MAPE",
        "SMAPE",
        "RMSPE",
        "MTD_p2",
        "EMLAE",
        "MALE",
        "MAE",
        "RMSE",
        "MedAE",
        "MTD_p1",
        "MSE",
        "MSLE",
        "VAR",
        "R2",
        "PR",
        "SR",
        "KT",
        "WT",
        "MGC",
    ]
    metrics = pd.Series(data=metrics_values, index=metrics_names, name="评估指标值")
    metrics = metrics.round(4)
    metrics.to_excel(
        output_path + f"\{i}_第一次决策的20种评估指标计算结果.xlsx",
        index=True,
        encoding="utf-8-sig",
        sheet_name=f"{i}20种评估指标的取值",
    )
    print(f"metrics: \n {metrics}", "\n")

    # 作图比较原始销量和预测销量，以及预测销量的置信区间，并输出保存图片
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(
        sm_qielei_seg["销售日期"][-output_index:],
        sm_qielei_seg["原始销量"][-output_index:],
        label="原始销量",
        marker="o",
    )
    ax.plot(
        sm_qielei_seg["销售日期"][-output_index:],
        sm_qielei_seg["预测销量"][-output_index:],
        label="预测销量",
        marker="o",
    )
    ax.fill_between(
        sm_qielei_seg["销售日期"][-output_index:],
        forecast_amount["yhat_lower"][-output_index:],
        forecast_amount["yhat_upper"][-output_index:],
        color="grey",
        alpha=0.2,
        label=f"{int(interval_width*100)}%的置信区间",
    )
    ax.set_xlabel("销售日期")
    ax.set_ylabel("销量")
    ax.set_title(f"{i}预测期第一次订货量时序对比图")
    ax.legend()
    plt.savefig(output_path + f"\{i}_forecast.svg", dpi=300, bbox_inches="tight")
    plt.show()

    # 第二轮订货定价优化
    qielei_prophet_price = sm_qielei[["busdate", "sum_price"]].rename(
        columns={"busdate": "ds", "sum_price": "y"}
    )
    m_price = Prophet(
        seasonality_mode="multiplicative",
        holidays_prior_scale=10,
        seasonality_prior_scale=10,
        mcmc_samples=mcmc_samples,
        interval_width=interval_width,
    )
    m_price.add_country_holidays(country_name="CN")
    m_price.add_seasonality(name="weekly", period=7, fourier_order=10, prior_scale=10)
    m_price.add_seasonality(name="yearly", period=365, fourier_order=3, prior_scale=10)
    m_price.fit(qielei_prophet_price)
    future_price = m_price.make_future_dataframe(periods=periods)
    forecast_price = m_price.predict(future_price)
    fig1 = m_price.plot(forecast_price)
    plt.show()
    fig2 = m_price.plot_components(forecast_price)
    plt.show()
    fig1.savefig(output_path + f"\{i}_fit_revenue.svg", dpi=300, bbox_inches="tight")
    fig2.savefig(
        output_path + f"\{i}_fit_revenue_components.svg", dpi=300, bbox_inches="tight"
    )
    # if the "yhat" column in forecast_price <= 0, then replace it with the mean of last 7 days' "yhat" before it
    forecast_price["yhat"] = forecast_price["yhat"].apply(
        lambda x: (
            max(
                forecast_price["yhat"].rolling(7, min_periods=1).mean().iloc[-1],
                np.random.uniform(0, max(forecast_price["yhat"].median(), 0.1)),
            )
            if x < 0
            else x
        )
    )

    sm_qielei["unit_cost"] = sm_qielei["sum_cost"] / sm_qielei["amount"]
    qielei_prophet_cost = sm_qielei[["busdate", "unit_cost"]].rename(
        columns={"busdate": "ds", "unit_cost": "y"}
    )
    m_cost = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode="multiplicative",
        holidays_prior_scale=10,
        seasonality_prior_scale=10,
        mcmc_samples=mcmc_samples,
        interval_width=interval_width,
    )
    m_cost.add_country_holidays(country_name="CN")
    m_cost.fit(qielei_prophet_cost)
    future_cost = m_cost.make_future_dataframe(periods=periods)
    forecast_cost = m_cost.predict(future_cost)
    fig1 = m_cost.plot(forecast_cost)
    plt.show()
    fig2 = m_cost.plot_components(forecast_cost)
    plt.show()
    fig1.savefig(output_path + f"\{i}_fit_unit_cost.svg", dpi=300, bbox_inches="tight")
    fig2.savefig(
        output_path + f"\{i}_fit_unit_cost_components.svg", dpi=300, bbox_inches="tight"
    )
    forecast_cost.to_excel(
        output_path + f"\{i}_单位成本时序分解.xlsx",
        index=False,
        encoding="utf-8-sig",
        sheet_name=f"{i}预测单位成本",
    )

    forecast = forecast_price[["ds", "yhat"]][-periods:]
    forecast["price"] = forecast["yhat"] / q_star
    if (
        forecast["price"].mean()
        > acct_ori["price"].mean() + 3 * acct_ori["price"].std()
    ) or (
        forecast["price"].mean()
        <= max(0, acct_ori["price"].mean() - 3 * acct_ori["price"].std())
    ):
        forecast["price"] = (
            acct_ori["price"].mean()
            + np.random.randn(len(forecast["price"]))
            * acct_ori["price"].std()
            / acct_ori["price"].mean()
        )
    forecast["unit_cost"] = forecast_cost["yhat"][-periods:]
    if (
        forecast["unit_cost"].mean()
        > acct_ori["unit_cost"].mean() + 3 * acct_ori["unit_cost"].std()
    ) or (
        forecast["unit_cost"].mean()
        <= max(0, acct_ori["unit_cost"].mean() - 3 * acct_ori["unit_cost"].std())
    ):
        forecast["unit_cost"] = (
            acct_ori["unit_cost"].mean()
            + np.random.randn(len(forecast["unit_cost"]))
            * acct_ori["unit_cost"].std()
            / acct_ori["unit_cost"].mean()
        )
    forecast["profit"] = (forecast["price"] - forecast["unit_cost"]) / forecast["price"]
    if (
        forecast["profit"].mean()
        > acct_ori["profit"].mean() + 3 * acct_ori["profit"].std()
    ) or (
        forecast["profit"].mean()
        <= max(0, acct_ori["profit"].mean() - 3 * acct_ori["profit"].std())
    ):
        forecast["profit"] = (
            acct_ori["profit"].mean()
            + np.random.randn(len(forecast["profit"]))
            * acct_ori["profit"].std()
            / acct_ori["profit"].mean()
        )
    # 将forecast['profit']中<=0的值，替换为平均毛利率
    forecast["profit"] = forecast["profit"].apply(
        lambda x: max(profit_avg, 0.01) if x <= 0 else x
    )

    # 用newsvendor模型计算sm_qielei_amt_ext的平稳订货量q_steady
    f_star = fitter.Fitter(sm_qielei_amt_ext, distributions="gamma")
    f_star.fit()
    print(
        f'拟合分布的最优参数是: \n {np.array(f_star.fitted_param["gamma"]).round(4)}',
        "\n",
    )
    q_steady_star = []
    for j in range(len(forecast["profit"])):
        q_steady_star.append(
            stats.gamma.ppf(forecast["profit"].values[j], *f_star.fitted_param["gamma"])
        )
    q_steady_star = np.array(q_steady_star)
    print(f"q_steady_star = {q_steady_star.round(3)}", "\n")
    pd.Series(f_star.fitted_param["gamma"]).to_excel(
        output_path + f"\{i}_第二次定价订货的最优分布参数.xlsx",
        index=True,
        encoding="utf-8-sig",
        sheet_name=f"{i}第二次定价订货的最优分布参数",
    )

    all_set["total_effect"] = all_set[
        ["holiday_effect", "weekly_effect_avg", "yearly_effect_avg"]
    ].sum(axis=1)
    q_star_new = q_steady_star * (1 + all_set["total_effect"][-periods:])
    print(f"q_star_new =\n{q_star_new.round(3)}", "\n")
    forecast["加载毛利率时间效应的第二次报童订货量"] = q_steady_star
    forecast["q_star_new"] = q_star_new
    forecast.rename(
        columns={
            "ds": "销售日期",
            "yhat": "预测金额",
            "price": "预测单价",
            "unit_cost": "预测成本单价",
            "profit": "预测毛利率",
            "q_star_new": "加载销量时间效应的最终订货量",
        },
        inplace=True,
    )
    forecast[["预测金额", "预测单价", "预测成本单价"]] = forecast[
        ["预测金额", "预测单价", "预测成本单价"]
    ].applymap(lambda x: round(x, 2))
    forecast[
        ["加载毛利率时间效应的第二次报童订货量", "加载销量时间效应的最终订货量"]
    ] = forecast[
        ["加载毛利率时间效应的第二次报童订货量", "加载销量时间效应的最终订货量"]
    ].apply(
        lambda x: round(x).astype(int)
    )
    forecast["预测毛利率"] = forecast["预测毛利率"].apply(lambda x: round(x, 3))
    forecast_output = forecast.iloc[:output_index]
    forecast_output["销售日期"] = forecast_output["销售日期"].dt.date

    # 用均方根求amount_base和forecast_output['加载销量时间效应的最终订货量']的平均数
    forecast_output["加载销量时间效应的最终订货量"] = pd.Series(
        np.sqrt(
            (amount_base**2 + forecast_output["加载销量时间效应的最终订货量"] ** 2) / 2
        )
        + np.random.rand(len(forecast_output["加载销量时间效应的最终订货量"]))
        * 1
        / 10
        * acct_ori["amount"].std()
    ).astype(int)
    forecast_output["预测金额"] = (
        forecast_output["预测单价"] * forecast_output["加载销量时间效应的最终订货量"]
    )
    forecast_output["预测毛利率"] = (
        forecast_output["预测单价"] - forecast_output["预测成本单价"]
    ) / forecast_output["预测单价"]
    forecast_output["预测毛利率"] = forecast_output["预测毛利率"].apply(
        lambda x: round(x, 4) * 100
    )
    forecast_output.rename(columns={"预测毛利率": "预测毛利率(%)"}, inplace=True)

    forecast_output.to_excel(
        output_path
        + f"\{i}_在预测期每日的预测销售额、预测单价、预测成本、预测毛利率、加载毛利率时间效应的第二次报童订货量和加载销量时间效应的最终订货量.xlsx",
        index=False,
        encoding="utf-8-sig",
        sheet_name=f"问题2最终结果：{i}在预测期每日的预测销售额、预测单价、预测成本、预测毛利率、加载毛利率时间效应的第二次报童订货量和加载销量时间效应的最终订货量",
    )
    forecast_price.loc[forecast_price[-periods:].index, "yhat"] = forecast_output[
        "预测金额"
    ].values
    forecast_price.to_excel(
        output_path + f"\{i}_销售额时序分解.xlsx",
        index=False,
        encoding="utf-8-sig",
        sheet_name=f"{i}预测销售额时序分解",
    )

    # 评估指标
    res_new = ref.regression_evaluation_single(
        y_true=sm_qielei_all["实际销量"][-periods:].values,
        y_pred=forecast_output["加载销量时间效应的最终订货量"][-periods:].values,
    )
    accu_sin_new = ref.accuracy_single(
        y_true=sm_qielei_all["实际销量"][-periods:].values,
        y_pred=forecast_output["加载销量时间效应的最终订货量"][-periods:].values,
    )
    metrics_values_new = [accu_sin_new] + list(res_new[:-2])
    metrics_new = pd.Series(
        data=metrics_values_new, index=metrics_names, name="新评估指标值"
    )
    metrics_new = metrics_new.round(4)
    metrics_new.to_excel(
        output_path + f"\{i}_加载销量时间效应的最终订货量的20种评估指标计算结果.xlsx",
        index=True,
        encoding="utf-8-sig",
        sheet_name=f"{i}加载销量时间效应的最终订货量的评估指标值",
    )
    print(f"metrics_new: \n {metrics_new}", "\n")

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(
        sm_qielei_all["销售日期"][-output_index:],
        sm_qielei_all["原始销量"][-output_index:],
        marker="o",
        label="原始销量",
    )
    ax.plot(
        forecast_output["销售日期"][-output_index:],
        forecast_output["加载销量时间效应的最终订货量"][-output_index:],
        marker="o",
        label="加载销量时间效应的最终订货量",
    )
    ax.fill_between(
        sm_qielei_all["销售日期"][-output_index:],
        forecast_price["yhat_lower"][-output_index:] / forecast["预测单价"],
        forecast_price["yhat_upper"][-output_index:] / forecast["预测单价"],
        color="grey",
        alpha=0.2,
        label=f"{int(interval_width*100)}%的置信区间",
    )
    ax.set_xlabel("销售日期")
    ax.set_ylabel("销量")
    ax.set_title(f"{i}预测期加载销量时间效应的最终订货量对比图")
    ax.legend()
    plt.savefig(output_path + f"\{i}_forecast_final.svg", dpi=300, bbox_inches="tight")
    plt.show()

print("question_2运行完毕！", "\n\n\n")
