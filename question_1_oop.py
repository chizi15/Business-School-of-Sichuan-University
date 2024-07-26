# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as ss
import fitter
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os
from data_output import output_path_self_use, first_day, last_day

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 8)


def ts_dist_sm(name, data):

    # 在df_p1中，对各个sm_sort分别画时间序列图，横坐标是busdate，纵坐标是amount
    fig = plt.figure(figsize=(20, 10))
    plt.plot(data["busdate"], data["amount"])
    plt.title(f"{name}")
    plt.show()
    fig.savefig(
        output_path + "小分类_%s_销量时序.svg" % name
    )  # 按小分类聚合后的平均销量和平均价格
    fig.clear()

    # 对销量序列进行分布拟合比较
    f = fitter.Fitter(data["amount"], distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f"\n{comparison_of_distributions_qielei.round(4)}\n")
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    comparison_of_distributions_qielei.to_excel(
        output_path + f"小分类_{name}_comparison_of_distributions.xlsx",
        sheet_name=f"{name}_comparison of distributions",
    )

    # 给figure添加label和title，并保存输出对比分布图
    name_dist = list(f.get_best().keys())[0]
    print(f"best distribution: {name_dist}" "\n")
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f"{name}_销量分布拟合对比")
    plt.ylabel("Probability")
    plt.title(f"{name}_comparison of distributions")
    plt.show()
    figure.savefig(output_path + f"小分类_{name}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    # 绘制并保存输出最优分布图
    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, "b-.", label="f.y")
    plt.plot(f.x, f.fitted_pdf[name_dist], "r-", label="f.fitted_pdf")
    plt.xlabel(f"{name}_销量最优分布拟合")
    plt.ylabel("Probability")
    plt.title(f"best distribution: {name_dist}")
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"小分类_{name}_best distribution.svg")
    figure.clear()


def grouping_heatmap(sale_sm, coef, corr_neg):
    # 对数变换增强正态性，以加强对相关系数计算假设条件的满足程度
    sale_sm["amount"] = sale_sm["amount"].apply(lambda x: np.log1p(x))
    sale_sm["price"] = sale_sm["price"].apply(lambda x: np.log1p(x))
    # 筛选销量与价格负相关性强的小分类
    typeA = []
    typeB = []
    for code, data in sale_sm.groupby(["sm_sort_name"]):
        if len(data) > 5:
            r = ss.spearmanr(data["amount"], data["price"]).correlation
            if r < corr_neg:
                typeA.append(code)
            else:
                typeB.append(code)
    # 对sale_sm['amount']和price做np.log1p的逆变换，使数据回到原来的尺度
    sale_sm["amount"] = sale_sm["amount"].apply(lambda x: np.expm1(x))
    sale_sm["price"] = sale_sm["price"].apply(lambda x: np.expm1(x))
    sale_sm_a = sale_sm[sale_sm["sm_sort_name"].isin(typeA)]
    sale_sm_b = sale_sm[sale_sm["sm_sort_name"].isin(typeB)]
    print(
        f'销量与价格的负相关性强(小于{corr_neg})的小分类一共有{sale_sm_a["sm_sort_name"].nunique()}个'
    )
    print(
        f'销量与价格的负相关性弱(大于等于{corr_neg})的小分类一共有{sale_sm_b["sm_sort_name"].nunique()}个',
        "\n",
    )
    sale_sm_a.to_excel(
        output_path
        + f"小分类_销售数据_销量与价格的负相关性强(小于{corr_neg})的一组.xlsx"
    )  # 按小分类聚合后的平均销量和平均价格
    sale_sm_b.to_excel(
        output_path
        + f"小分类_销售数据_销量与价格的负相关性弱(大于等于{corr_neg})的一组.xlsx"
    )  # 按小分类聚合后的平均销量和平均价格

    # 计算负相关性强的小分类序列的相关系数并画热力图。
    # 先对df行转列
    sale_sm_a_t = pd.pivot(
        sale_sm_a, index="busdate", columns="sm_sort_name", values="amount"
    )
    # 计算每列间的相关性
    sale_sm_a_coe = sale_sm_a_t.corr(
        method="pearson"
    )  # Compute pairwise correlation of columns, excluding NA/null values
    # 画相关系数矩阵的热力图，并保存输出，每个小分类的名字都显示出来，排列稀疏
    plt.figure(figsize=(20, 20))
    sns.heatmap(sale_sm_a_coe, annot=True, xticklabels=True, yticklabels=True)
    plt.savefig(
        output_path
        + "小分类_销量与价格负相关性强的一组中，各个小分类销量间的corr_heatmap.svg"
    )  # 按小分类聚合后的平均销量和平均价格

    # 对typeA中小分类按相关系数的排序进行分组
    # 选择相关性大于coef的组合
    groups = []
    idxs = sale_sm_a_coe.index.to_list()
    for idx, row in sale_sm_a_coe.iterrows():
        group = row[row > coef].index.to_list()
        groups.append(group)
    # 删除重复使用的小分类
    groups_ = []
    for group in groups:
        diff_group = []
        for idx in group:
            if idx in idxs:
                idxs.remove(idx)
            else:
                diff_group.append(idx)
        group = set(group) - set(diff_group)
        if group:
            groups_.append(group)
    print(f"进行相关性排序，并以相关系数大于{coef}为条件进行分组后的结果\n{groups_}")

    # 将groups_中的集合转换为列表
    groups_ = [list(group) for group in groups_]
    groups_.append(typeB)
    print(f"最终分组结果\n{groups_}")
    # 将groups_中的列表转换为df，索引为组号，列名为各个小分类名
    groups_df = pd.DataFrame(pd.Series(groups_), columns=["sm_sort_name"])
    groups_df["group"] = groups_df.index + 1
    # 改变列的顺序
    groups_df = groups_df[["group", "sm_sort_name"]]
    groups_df.to_excel(
        output_path + f"小分类_相关性分组结果：以相关系数大于{coef}为条件.xlsx",
        index=False,
        sheet_name="最后一组是销量对价格不敏感的，前面若干组是销量对价格敏感的",
    )  # 按小分类聚合后的平均销量和平均价格

    return groups_


def grouped_ts_dist_sm(i, data, groups_):

    # 对list_df_avg中每个df画时间序列图，横坐标是busdate，纵坐标是amount，图名从组1到组7依次命名
    fig = plt.figure(figsize=(20, 10))
    plt.plot(data["busdate"], data["amount"])
    plt.title(f"{groups_[i]}")
    plt.show()
    fig.savefig(
        output_path
        + f"小分类_{str(groups_[i]).replace('[', '(').replace(']', ')')}_按相关性分组合并后的小分类销量时序.svg"
    )  # 按小分类聚合后的平均销量
    fig.clear()

    # 对销量序列进行分布拟合比较
    f = fitter.Fitter(data["amount"], distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f"\n{comparison_of_distributions_qielei.round(4)}\n")
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    # 将groups_[i]中的小分类名转换为字符串，再替换异常符号，以便作为excel文件名和sheet_name表名
    groups_[i] = str(groups_[i])
    groups_[i] = groups_[i].replace("'", "").replace("[", "(").replace("]", ")")
    comparison_of_distributions_qielei.to_excel(
        output_path + f"小分类_{groups_[i]}_comparison_of_distributions.xlsx",
        sheet_name=f"{groups_[i]}_comparison of distributions",
    )

    # 给figure添加label和title，并保存输出对比分布图
    name_dist = list(f.get_best().keys())[0]
    print(f"best distribution: {name_dist}" "\n")
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f"{groups_[i]}_销量分布拟合对比")
    plt.ylabel("Probability")
    plt.title(f"{groups_[i]}_comparison of distributions")
    plt.show()
    figure.savefig(output_path + f"小分类_{groups_[i]}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    # 绘制并保存输出最优分布图
    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, "b-.", label="f.y")
    plt.plot(f.x, f.fitted_pdf[name_dist], "r-", label="f.fitted_pdf")
    plt.xlabel(f"{groups_[i]}_销量最优分布拟合")
    plt.ylabel("Probability")
    plt.title(f"best distribution: {name_dist}")
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"小分类_{groups_[i]}_best distribution.svg")
    figure.clear()


def ts_dist_code(name, data):

    # 在df_p1中，对各个sm_sort分别画时间序列图，横坐标是busdate，纵坐标是amount
    fig = plt.figure(figsize=(20, 10))
    plt.plot(data["busdate"], data["amount"])
    plt.title(f"{name}")
    plt.show()
    fig.savefig(
        output_path + "单品_%s_销量时序.svg" % name
    )  # 按小分类聚合后的平均销量和平均价格
    fig.clear()

    # 对销量序列进行分布拟合比较
    f = fitter.Fitter(data["amount"], distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f"\n{comparison_of_distributions_qielei.round(4)}\n")
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    comparison_of_distributions_qielei.to_excel(
        output_path + f"单品_{name}_comparison_of_distributions.xlsx",
        sheet_name=f"{name}_comparison of distributions",
    )

    # 给figure添加label和title，并保存输出对比分布图
    name_dist = list(f.get_best().keys())[0]
    print(f"best distribution: {name_dist}" "\n")
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f"{name}_销量分布拟合对比")
    plt.ylabel("Probability")
    plt.title(f"{name}_comparison of distributions")
    plt.show()
    figure.savefig(output_path + f"单品_{name}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    # 绘制并保存输出最优分布图
    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, "b-.", label="f.y")
    plt.plot(f.x, f.fitted_pdf[name_dist], "r-", label="f.fitted_pdf")
    plt.xlabel(f"{name}_销量最优分布拟合")
    plt.ylabel("Probability")
    plt.title(f"best distribution: {name_dist}")
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"单品_{name}_best distribution.svg")
    figure.clear()


def grouping_heatmap_code(sale_sm, coef, corr_neg):
    # 对数变换增强正态性，以加强对相关系数计算假设条件的满足程度
    sale_sm["amount"] = sale_sm["amount"].apply(lambda x: np.log1p(x))
    sale_sm["price"] = sale_sm["price"].apply(lambda x: np.log1p(x))
    # 筛选销量与价格负相关性强的小分类
    typeA = []
    typeB = []
    for code, data in sale_sm.groupby(["name"]):
        if len(data) > 5:
            r = ss.spearmanr(data["amount"], data["price"]).correlation
            if r < corr_neg:
                typeA.append(code)
            else:
                typeB.append(code)
    # 对sale_sm['amount']和price做np.log1p的逆变换，使数据回到原来的尺度
    sale_sm["amount"] = sale_sm["amount"].apply(lambda x: np.expm1(x))
    sale_sm["price"] = sale_sm["price"].apply(lambda x: np.expm1(x))
    sale_sm_a = sale_sm[sale_sm["name"].isin(typeA)]
    sale_sm_b = sale_sm[sale_sm["name"].isin(typeB)]
    print(
        f'销量与价格的负相关性强(小于{corr_neg})的单品一共有{sale_sm_a["name"].nunique()}个'
    )
    print(
        f'销量与价格的负相关性弱(大于等于{corr_neg})的单品一共有{sale_sm_b["name"].nunique()}个',
        "\n",
    )
    sale_sm_a.to_excel(
        output_path + f"单品_销售数据_销量与价格的负相关性强(小于{corr_neg})的一组.xlsx"
    )
    sale_sm_b.to_excel(
        output_path
        + f"单品_销售数据_销量与价格的负相关性弱(大于等于{corr_neg})的一组.xlsx"
    )

    # 计算负相关性强的单品序列的相关系数并画热力图。
    # 先对df行转列
    sale_sm_a_t = pd.pivot(sale_sm_a, index="busdate", columns="name", values="amount")
    # 计算每列间的相关性
    sale_sm_a_coe = sale_sm_a_t.corr(
        method="pearson"
    )  # Compute pairwise correlation of columns, excluding NA/null values
    plt.figure(figsize=(20, 20))
    sns.heatmap(sale_sm_a_coe, annot=True, xticklabels=True, yticklabels=True)
    plt.savefig(
        output_path
        + "单品_销量与价格负相关性强的一组中，各个单品销量间的corr_heatmap.svg"
    )

    # 对typeA中小分类按相关系数的排序进行分组
    # 选择相关性大于coef的组合
    groups = []
    idxs = sale_sm_a_coe.index.to_list()
    for idx, row in sale_sm_a_coe.iterrows():
        group = row[row > coef].index.to_list()
        groups.append(group)
    groups_ = []
    for group in groups:
        diff_group = []
        for idx in group:
            if idx in idxs:
                idxs.remove(idx)
            else:
                diff_group.append(idx)
        group = set(group) - set(diff_group)
        if group:
            groups_.append(group)
    print(
        f"进行相关性排序，并以相关系数大于{round(coef, 2)}为条件进行分组后的结果:\n{groups_}\n"
    )

    # 将groups_中的集合转换为列表
    groups_ = [list(group) for group in groups_]
    groups_.append(typeB)
    print(f"最终分组结果\n{groups_}")
    # 将groups_中的列表转换为df，索引为组号，列名为各个小分类名
    groups_df = pd.DataFrame(pd.Series(groups_), columns=["name"])
    groups_df["group"] = groups_df.index + 1
    # 改变列的顺序
    groups_df = groups_df[["group", "name"]]
    groups_df.to_excel(
        output_path + f"单品_相关性分组结果：以相关系数大于{coef}为条件.xlsx",
        index=False,
        sheet_name="最后一组是销量对价格不敏感的，前面若干组是销量对价格敏感的",
    )

    return groups_


def grouped_ts_dist_code(i, data, groups_):

    fig = plt.figure(figsize=(20, 10))
    plt.plot(data["busdate"], data["amount"])
    plt.title(f"{groups_[i]}")
    plt.show()
    fig.savefig(
        output_path
        + f"单品_{str(groups_[i]).replace('[', '(').replace(']', ')')}_按相关性分组合并后的销量时序.svg"
    )  # 按小分类聚合后的平均销量
    fig.clear()

    # 对销量序列进行分布拟合比较
    f = fitter.Fitter(data["amount"], distributions=distributions, timeout=10)
    f.fit()
    comparison_of_distributions_qielei = f.summary(Nbest=len(distributions))
    print(f"\n{comparison_of_distributions_qielei.round(4)}\n")
    comparison_of_distributions_qielei = comparison_of_distributions_qielei.round(4)
    # 将groups_[i]中的小分类名转换为字符串，再替换异常符号，以便作为excel文件名和sheet_name表名
    groups_[i] = str(groups_[i])
    groups_[i] = groups_[i].replace("'", "").replace("[", "(").replace("]", ")")
    comparison_of_distributions_qielei.to_excel(
        output_path + f"单品_{groups_[i]}_comparison_of_distributions.xlsx",
        sheet_name=f"{groups_[i]}_comparison of distributions",
    )

    # 给figure添加label和title，并保存输出对比分布图
    name_dist = list(f.get_best().keys())[0]
    print(f"best distribution: {name_dist}" "\n")
    figure = plt.gcf()  # 获取当前图像
    plt.xlabel(f"{groups_[i]}_销量分布拟合对比")
    plt.ylabel("Probability")
    plt.title(f"{groups_[i]}_comparison of distributions")
    plt.show()
    figure.savefig(output_path + f"单品_{groups_[i]}_comparison of distributions.svg")
    figure.clear()  # 先画图plt.show，再释放内存

    # 绘制并保存输出最优分布图
    figure = plt.gcf()  # 获取当前图像
    plt.plot(f.x, f.y, "b-.", label="f.y")
    plt.plot(f.x, f.fitted_pdf[name_dist], "r-", label="f.fitted_pdf")
    plt.xlabel(f"{groups_[i]}_销量最优分布拟合")
    plt.ylabel("Probability")
    plt.title(f"best distribution: {name_dist}")
    plt.legend()
    plt.show()
    figure.savefig(output_path + f"单品_{groups_[i]}_best distribution.svg")
    figure.clear()


if __name__ == "__main__":

    # 第0部分：设置全局变量及数据预处理
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
    input_path = output_path_self_use + "\\"
    output_path = r"D:\Work info\SCU\MathModeling\2023\data\processed\question_1" + "\\"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 读取数据
    df = pd.read_csv(input_path + "account.csv")
    df.sort_values(by=["busdate"], inplace=True)

    # 输出这三条时序图中，非空数据的起止日期，用循环实现
    for col in ["amount", "sum_cost", "sum_price"]:
        print(
            f'{col}非空数据的起止日期为：{df[df[col].notnull()]["busdate"].min()}到{df[df[col].notnull()]["busdate"].max()}',
            "\n",
        )

    # 断言df中数值型字段的起止日期相同
    assert (
        df[df["amount"].notnull()]["busdate"].min()
        == df[df["sum_cost"].notnull()]["busdate"].min()
        == df[df["sum_price"].notnull()]["busdate"].min()
    ), "三个字段非空数据的开始日期不相同"
    assert (
        df[df["amount"].notnull()]["busdate"].max()
        == df[df["sum_cost"].notnull()]["busdate"].max()
        == df[df["sum_price"].notnull()]["busdate"].max()
    ), "三个字段非空数据的结束日期不相同"

    sort = pd.read_csv(input_path + "commodity.csv")
    # 拼接账表和商品资料表
    df = pd.merge(df, sort, how="left", on=["code", "class"])
    df["busdate"] = pd.to_datetime(df["busdate"])
    df.drop(columns=["sum_disc"], inplace=True)

    # 第一部分：小分类层级
    # 计算小分类层级的分布和相关性分组
    coef = 0.5  # 相关系数排序分组时的阈值
    corr_neg = -0.3  # 销量与售价的负相关性阈值

    df_p1 = df
    print(df_p1.dtypes, "\n")
    print(sort.dtypes, "\n")
    df_p1 = (
        df_p1.groupby(["sm_sort", "busdate"])
        .agg({"amount": "mean", "sum_price": "mean", "sum_cost": "mean"})
        .reset_index()
    )
    df_p1 = df_p1.merge(
        sort.drop(columns=["class", "code", "name"]).drop_duplicates(),
        on="sm_sort",
        how="left",
    )

    # 计算平均售价、进价和毛利率
    df_p1["price"] = df_p1["sum_price"] / df_p1["amount"]
    df_p1["cost_price"] = df_p1["sum_cost"] / df_p1["amount"]
    df_p1["profit"] = (df_p1["price"] - df_p1["cost_price"]) / df_p1["price"]
    sale_sm = df_p1.dropna()
    sale_sm = sale_sm[sale_sm["profit"] >= 0]
    sale_sm.sort_values(by=["sm_sort", "busdate"], inplace=True)
    print(f'总共有{sale_sm["sm_sort"].nunique()}个sm_sort', "\n")
    # 判断sale_sm['sm_sort']中是否有小分类的名称中包含'.'，或者sale_sm['sm_sort']的数据类型是否为float64
    if (
        sale_sm["sm_sort"].dtype == "float64"
        or sale_sm["sm_sort"].astype(str).str.contains("\.").any()
    ):
        print("sale_sm['sm_sort'] is of type float64 or contains decimal points.")
        sale_sm["sm_sort"] = sale_sm["sm_sort"].astype(str).str.split(".").str[0]
    else:
        print(
            "sale_sm['sm_sort'] is not of type float64 and does not contain decimal points."
        )

    # # 绘制各个小分类的平均销量时序图，及其分布比较，并得到最优分布
    Parallel(n_jobs=-1, verbose=50)(
        delayed(ts_dist_sm)(name, data)
        for name, data in sale_sm.groupby(["sm_sort_name"])
    )

    groups_ = grouping_heatmap(sale_sm, coef, corr_neg)

    # 对groups_中的每个组，从df_p1中筛选出对应的数据，组成list_df
    list_df = [df_p1[df_p1["sm_sort_name"].isin(group)] for group in groups_]
    # 循环对list_df中每个df按busdate进行合并groupby，并求均值
    list_df_avg = [
        data.groupby(["busdate"])
        .agg({"amount": "mean", "sum_price": "mean", "sum_cost": "mean"})
        .reset_index()
        for data in list_df
    ]

    Parallel(n_jobs=-1, verbose=50)(
        delayed(grouped_ts_dist_sm)(i, data, groups_)
        for i, data in enumerate(list_df_avg)
    )

    print("\nquestion_1小分类运行完毕！", "\n\n")

    # 第二部分：单品层级
    # 计算单品层级的分布和相关性分组
    coef = round(1 / 3, 2)  # 相关系数排序分组时的阈值
    corr_neg = -0.2  # 销量与售价的负相关性阈值

    code_busdate = (
        df.groupby("code")
        .agg(min_busdate=("busdate", "min"), max_busdate=("busdate", "max"))
        .reset_index()
    )
    code_busdate_codes = code_busdate[
        (code_busdate["min_busdate"] == first_day)
        & (code_busdate["max_busdate"] == last_day)
    ]["code"]
    df_p1 = df[df["code"].isin(code_busdate_codes)]

    # 计算平均售价、进价和毛利率
    df_p1["price"] = df_p1["sum_price"] / df_p1["amount"]
    df_p1["cost_price"] = df_p1["sum_cost"] / df_p1["amount"]
    df_p1["profit"] = (df_p1["price"] - df_p1["cost_price"]) / df_p1["price"]
    sale_sm = df_p1.dropna()
    sale_sm = sale_sm[sale_sm["profit"] >= 0]
    sale_sm.sort_values(by=["code", "busdate"], inplace=True)
    print(f'总共有{sale_sm["code"].nunique()}个codes', "\n")
    # 判断sale_sm['code']中是否有小分类的名称中包含'.'，或者sale_sm['code']的数据类型是否为float64
    if (
        sale_sm["code"].dtype == "float64"
        or sale_sm["code"].astype(str).str.contains("\.").any()
    ):
        print("sale_sm['code'] is of type float64 or contains decimal points.")
        sale_sm["code"] = sale_sm["code"].astype(str).str.split(".").str[0]
    else:
        print(
            "sale_sm['code'] is not of type float64 and does not contain decimal points."
        )

    # 绘制各个单品的平均销量时序图，及其分布比较，并得到最优分布
    Parallel(n_jobs=-1, verbose=50)(
        delayed(ts_dist_code)(name, data) for name, data in sale_sm.groupby(["name"])
    )
    print("\nts_dist_code parallel done!\n")

    groups_ = grouping_heatmap_code(sale_sm, coef, corr_neg)

    # 对groups_中的每个组，从df_p1中筛选出对应的数据，组成list_df
    list_df = [df_p1[df_p1["name"].isin(group)] for group in groups_]
    # 循环对list_df中每个df按busdate进行合并groupby，并求均值
    list_df_avg = [
        data.groupby(["busdate"])
        .agg({"amount": "mean", "sum_price": "mean", "sum_cost": "mean"})
        .reset_index()
        for data in list_df
    ]

    # 对list_df_avg中每个df画时间序列图，横坐标是busdate，纵坐标是amount
    Parallel(n_jobs=-1, verbose=50)(
        delayed(grouped_ts_dist_code)(i, data, groups_)
        for i, data in enumerate(list_df_avg)
    )

    print("\n\nquestion_1单品层级运行完毕！", "\n")
