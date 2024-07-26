# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 6)


input_path = (
    r"D:\Work info\WestUnion\data\processed\HLJ\脱敏及筛选后样本数据\output" + "\\"
)
output_path = r"D:\Work info\SCU\MathModeling\2023\data\output" + "\\"
output_path_self_use = (
    r"D:\Work info\SCU\MathModeling\2023\data\ZNEW_DESENS\ZNEW_DESENS\sampledata" + "\\"
)
output_path_match = (
    r"D:\Work info\SCU\MathModeling\2023\data\processed" + "\\" + "july_first" + "\\"
)
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path_self_use, exist_ok=True)
os.makedirs(output_path_match, exist_ok=True)

first_day = "2020-07-01"
last_day = "2023-07-01"
sm_sort_name = ["食用菌", "花叶类", "水生根茎类", "辣椒类", "茄类", "花菜类"]
unit_cost_critical = 0  # 进货单价的筛选阈值，小于等于该值的数据将被剔除


if __name__ == "__main__":

    if last_day == "2023-06-30":
        code_sm = pd.read_excel(f"{input_path}" + "附件1-单品-分类.xlsx")
        code_sm[["单品编码", "分类编码"]] = code_sm[["单品编码", "分类编码"]].astype(
            str
        )
        print(
            f"code_sm['单品编码'].nunique(): {code_sm['单品编码'].nunique()}\ncode_sm['单品名称'].nunique(): {code_sm['单品名称'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\n"
        )
        run_code = pd.read_excel(f"{input_path}" + "附件2-流水-销量-售价.xlsx")
        run_code["单品编码"] = run_code["单品编码"].astype(str)
        print(f"run_code['单品编码'].nunique(): {run_code['单品编码'].nunique()}\n")

        # 将code_sm中有，但run_code中没有的单品编码筛选出来
        code_sm_not_in_run_code = code_sm[
            ~code_sm["单品编码"].isin(run_code["单品编码"])
        ]
        print("附件1中有，附件2中没有的单品编码：\n", code_sm_not_in_run_code, "\n")
        code_sm_not_in_run_code.to_excel(
            f"{output_path}/附件1中有但附件2中没有的单品编码.xlsx", index=False
        )

        acct_code = (
            run_code.groupby(["单品编码", "销售日期"])["销量(千克)"].sum().reset_index()
        )
        acct_com = pd.merge(acct_code, code_sm, on="单品编码", how="left")
        pd.set_option("display.max_rows", 10)
        print(acct_com.dtypes, "\n")
        print(acct_com.isnull().sum(), "\n")
        pd.set_option("display.max_rows", 6)
        acct_com_sm = (
            acct_com.groupby(["分类编码", "分类名称", "销售日期"])["销量(千克)"]
            .sum()
            .reset_index()
        )

        # 将acct_com_sm中的分类编码和分类名称两列合并，形成新的分类编码列，并用_连接
        acct_com_sm["分类编码_名称"] = (
            acct_com_sm["分类编码"] + "_" + acct_com_sm["分类名称"]
        )
        acct_com_sm.drop(columns=["分类编码", "分类名称"], inplace=True)
        # 按分类编码_名称列的不同取值，对销售日期列的值进行分组，形成新的列
        acct_com_sm = acct_com_sm.pivot_table(
            index="销售日期",
            columns="分类编码_名称",
            values="销量(千克)",
            aggfunc=np.sum,
        )
        # 将销售日期列的数据类型转换为字符串型，不带时分秒
        acct_com_sm.index = acct_com_sm.index.astype(str)
        acct_com_sm.to_excel(f"{output_path}/分类日汇总销售.xlsx", index=True)

        acct_com.drop(columns=["分类编码", "分类名称"], inplace=True)
        acct_com["单品编码_名称"] = acct_com["单品编码"] + "_" + acct_com["单品名称"]
        acct_com.drop(columns=["单品编码", "单品名称"], inplace=True)
        acct_com = acct_com.pivot_table(
            index="销售日期",
            columns="单品编码_名称",
            values="销量(千克)",
            aggfunc=np.sum,
        )
        acct_com.index = acct_com.index.astype(str)
        acct_com.to_excel(f"{output_path}/单品日汇总销售.xlsx", index=True)

        commodity = pd.read_csv(f"{input_path}commodity.csv")
        # 先转成int64，以免位数超限被转换为负数
        if not isinstance(commodity["code"].iloc[0], str):
            commodity[["code", "sm_sort", "md_sort", "bg_sort"]] = (
                commodity[["code", "sm_sort", "md_sort", "bg_sort"]]
                .astype("Int64")
                .astype(str)
            )
        # commodity按sm_sort_name进行第一次筛选
        commodity = commodity[commodity["sm_sort_name"].isin(sm_sort_name)]
        commodity = commodity[
            ~(
                (commodity["sm_sort_name"] == "茄类")
                & (
                    (commodity["name"].str.contains("番茄"))
                    | (commodity["name"].str.contains("西红柿"))
                )
            )
        ]

        account = pd.read_csv(f"{input_path}account.csv")
        # 判断account中code列的数据类型是否为str，如果不是，则转换为str
        if not isinstance(account["code"].iloc[0], str):
            account["code"] = account["code"].astype("Int64").astype(str)
        # account按commodity中的code进行第一次筛选
        account = account[account["code"].isin(commodity["code"])]
        # 将account中busdate列的数据类型转换为日期类型，但不带时分秒
        account["busdate"] = pd.to_datetime(account["busdate"], format="%Y-%m-%d")
        account.sort_values(by=["busdate", "code"], inplace=True)
        # account按日期范围进行第二次筛选
        account = account[
            (account["busdate"] >= first_day) & (account["busdate"] <= last_day)
        ]
        account["busdate"] = account["busdate"].apply(lambda x: x.date())
        account["unit_cost"] = account["sum_cost"] / account["amount"]
        account.dropna(subset=["unit_cost"], inplace=True)
        account["unit_cost"] = account["unit_cost"].round(2)
        # account按unit_cost列进行第三次筛选。以此账表中的code和busdate，作为后续筛选commodity和running的基准。
        account = account[account["unit_cost"] > unit_cost_critical]
        print(f"account.isnull().sum():\n{account.isnull().sum().T}", "\n")
        print(account.info(), "\n")

        account.to_csv(f"{output_path_self_use}/account.csv", index=False)
        account.rename(
            columns={
                "class": "课别",
                "code": "单品编码",
                "busdate": "日期",
                "unit_cost": "当天进货单价(元)",
            },
            inplace=True,
        )
        account.drop(
            columns=["organ", "sum_cost", "amount", "sum_price", "sum_disc"],
            inplace=True,
        )
        account.to_excel(f"{output_path}/account.xlsx", index=False)

        # account中code形成基准后，再次筛选commodity，才能输出，使得commodity中的code与account中的code一致
        commodity = commodity[commodity["code"].isin(account["单品编码"])]
        print(f"commodity.isnull().sum():\n{commodity.isnull().sum()}", "\n")
        print("commodity.info()", "\n", commodity.info(), "\n")

        commodity.to_csv(f"{output_path_self_use}/commodity.csv", index=False)
        commodity.rename(
            columns={
                "class": "课别",
                "code": "单品编码",
                "name": "单品名称",
                "sm_sort": "小分类编码",
                "md_sort": "中分类编码",
                "bg_sort": "大分类编码",
                "sm_sort_name": "小分类名称",
                "md_sort_name": "中分类名称",
                "bg_sort_name": "大分类名称",
            },
            inplace=True,
        )
        commodity.to_excel(f"{output_path}/commodity.xlsx", index=False)

        running = pd.read_csv(f"{input_path}running.csv")
        if not isinstance(running["code"].iloc[0], str):
            running["code"] = running["code"].astype("Int64").astype(str)
        # running按最终形成基准的commodity中的code进行第一次筛选
        running = running[running["code"].isin(commodity["单品编码"])]
        running["selldate"] = pd.to_datetime(running["selldate"])
        running["selldate"] = running["selldate"].apply(lambda x: x.date())
        # 将running中selldate和code，与account中日期和单品编码相同的筛选出来
        running = running.merge(
            account[["日期", "单品编码"]],
            how="inner",
            left_on=["selldate", "code"],
            right_on=["日期", "单品编码"],
        )
        # merge之后马上剔除多余的列，不能留到后面有相同列名的时候一起剔除，否则会剔除掉account中的日期和单品编码
        running.drop(columns=["日期", "单品编码"], inplace=True)
        running.sort_values(by=["selldate", "code"], inplace=True)
        running["打折销售"] = ["是" if x > 0 else "否" for x in running["sum_disc"]]
        assert (
            running["打折销售"].value_counts().values.sum() == running.shape[0]
        ), "流水表打折销售列计算有误"

        # 如果苹果在commodity的小分类名称中存在，需要输出running表，用于question_3_pre.py
        if "苹果" in commodity["小分类名称"].unique():
            running.to_csv(f"{output_path_self_use}/running.csv", index=False)

        running.rename(
            columns={
                "selldate": "销售日期",
                "selltime": "扫码销售时间",
                "class": "课别",
                "code": "单品编码",
                "amount": "销量",
                "price": "销售单价(元)",
                "type": "销售类型",
            },
            inplace=True,
        )
        running.drop(columns=["organ", "sum_disc", "sum_sell"], inplace=True)

        run_com = pd.merge(running, commodity, on=["课别", "单品编码"], how="left")
        print(run_com["小分类名称"].value_counts().sort_values(ascending=False), "\n")
        print(
            f"小分类编码与名称不唯一匹配的个数：{sum(run_com['小分类编码'].value_counts().sort_values(ascending=False).values != run_com['小分类名称'].value_counts().sort_values(ascending=False).values)}",
            "\n",
        )
        print(f"running.isnull().sum():\n{running.isnull().sum()}", "\n")
        print("running.info()", "\n", running.info(), "\n")

        try:
            running.to_excel(f"{output_path}/running.xlsx", index=False)
        except:
            running.to_csv(
                f"{output_path}/running.csv", index=False, encoding="utf-8-sig"
            )  # encoding='utf-8-sig'，解决excel打开，中文是乱码的问题
        print(running["销售类型"].value_counts().sort_values(ascending=False), "\n")
        print(running["打折销售"].value_counts().sort_values(ascending=False), "\n")

        print("data_output.py运行完毕！")

    elif last_day == "2023-07-01":

        code_sm = pd.read_excel(f"{input_path}" + "附件1-单品-分类.xlsx")
        code_sm[["单品编码", "分类编码"]] = code_sm[["单品编码", "分类编码"]].astype(
            str
        )
        print(
            f"code_sm['单品编码'].nunique(): {code_sm['单品编码'].nunique()}\ncode_sm['单品名称'].nunique(): {code_sm['单品名称'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\ncode_sm['分类编码'].nunique(): {code_sm['分类编码'].nunique()}\n"
        )
        run_code = pd.read_excel(f"{input_path}" + "附件2-流水-销量-售价.xlsx")
        run_code["单品编码"] = run_code["单品编码"].astype(str)
        print(f"run_code['单品编码'].nunique(): {run_code['单品编码'].nunique()}\n")

        # 将code_sm中有，但run_code中没有的单品编码筛选出来
        code_sm_not_in_run_code = code_sm[
            ~code_sm["单品编码"].isin(run_code["单品编码"])
        ]
        print("附件1中有，附件2中没有的单品编码：\n", code_sm_not_in_run_code, "\n")
        code_sm_not_in_run_code.to_excel(
            f"{output_path}/附件1中有但附件2中没有的单品编码.xlsx", index=False
        )

        acct_code = (
            run_code.groupby(["单品编码", "销售日期"])["销量(千克)"].sum().reset_index()
        )
        acct_com = pd.merge(acct_code, code_sm, on="单品编码", how="left")
        pd.set_option("display.max_rows", 10)
        print(acct_com.dtypes, "\n")
        print(acct_com.isnull().sum(), "\n")
        pd.set_option("display.max_rows", 6)
        acct_com_sm = (
            acct_com.groupby(["分类编码", "分类名称", "销售日期"])["销量(千克)"]
            .sum()
            .reset_index()
        )

        # 将acct_com_sm中的分类编码和分类名称两列合并，形成新的分类编码列，并用_连接
        acct_com_sm["分类编码_名称"] = (
            acct_com_sm["分类编码"] + "_" + acct_com_sm["分类名称"]
        )
        acct_com_sm.drop(columns=["分类编码", "分类名称"], inplace=True)
        # 按分类编码_名称列的不同取值，对销售日期列的值进行分组，形成新的列
        acct_com_sm = acct_com_sm.pivot_table(
            index="销售日期",
            columns="分类编码_名称",
            values="销量(千克)",
            aggfunc=np.sum,
        )
        # 将销售日期列的数据类型转换为字符串型，不带时分秒
        acct_com_sm.index = acct_com_sm.index.astype(str)
        acct_com_sm.to_excel(f"{output_path}/分类日汇总销售.xlsx", index=True)

        acct_com.drop(columns=["分类编码", "分类名称"], inplace=True)
        acct_com["单品编码_名称"] = acct_com["单品编码"] + "_" + acct_com["单品名称"]
        acct_com.drop(columns=["单品编码", "单品名称"], inplace=True)
        acct_com = acct_com.pivot_table(
            index="销售日期",
            columns="单品编码_名称",
            values="销量(千克)",
            aggfunc=np.sum,
        )
        acct_com.index = acct_com.index.astype(str)
        acct_com.to_excel(f"{output_path}/单品日汇总销售.xlsx", index=True)

        commodity = pd.read_csv(f"{input_path}commodity.csv")
        # 先转成int64，以免位数超限被转换为负数
        if not isinstance(commodity["code"].iloc[0], str):
            commodity[["code", "sm_sort", "md_sort", "bg_sort"]] = (
                commodity[["code", "sm_sort", "md_sort", "bg_sort"]]
                .astype("Int64")
                .astype(str)
            )
        # commodity按sm_sort_name进行第一次筛选
        commodity = commodity[commodity["sm_sort_name"].isin(sm_sort_name)]
        commodity = commodity[
            ~(
                (commodity["sm_sort_name"] == "茄类")
                & (
                    (commodity["name"].str.contains("番茄"))
                    | (commodity["name"].str.contains("西红柿"))
                )
            )
        ]

        account = pd.read_csv(f"{input_path}account.csv")
        # 判断account中code列的数据类型是否为str，如果不是，则转换为str
        if not isinstance(account["code"].iloc[0], str):
            account["code"] = account["code"].astype("Int64").astype(str)
        # account按commodity中的code进行第一次筛选
        account = account[account["code"].isin(commodity["code"])]
        # 将account中busdate列的数据类型转换为日期类型，但不带时分秒
        account["busdate"] = pd.to_datetime(account["busdate"], format="%Y-%m-%d")
        account.sort_values(by=["busdate", "code"], inplace=True)
        # account按日期范围进行第二次筛选
        account = account[
            (account["busdate"] >= first_day) & (account["busdate"] <= last_day)
        ]
        account["busdate"] = account["busdate"].apply(lambda x: x.date())
        account["unit_cost"] = account["sum_cost"] / account["amount"]
        account.dropna(subset=["unit_cost"], inplace=True)
        account["unit_cost"] = account["unit_cost"].round(2)
        # account按unit_cost列进行第三次筛选。以此账表中的code和busdate，作为后续筛选commodity和running的基准。
        account = account[account["unit_cost"] > unit_cost_critical]
        print(f"account.isnull().sum():\n{account.isnull().sum().T}", "\n")
        print(account.info(), "\n")

        account.to_csv(f"{output_path_self_use}/account.csv", index=False)
        account.rename(
            columns={
                "class": "课别",
                "code": "单品编码",
                "busdate": "日期",
                "unit_cost": "当天进货单价(元)",
            },
            inplace=True,
        )
        account.drop(
            columns=["organ", "sum_cost", "amount", "sum_price", "sum_disc"],
            inplace=True,
        )
        account.to_excel(f"{output_path}/account.xlsx", index=False)

        # account中code形成基准后，再次筛选commodity，才能输出，使得commodity中的code与account中的code一致
        commodity = commodity[commodity["code"].isin(account["单品编码"])]
        print(f"commodity.isnull().sum():\n{commodity.isnull().sum()}", "\n")
        print("commodity.info()", "\n", commodity.info(), "\n")

        commodity.to_csv(f"{output_path_self_use}/commodity.csv", index=False)
        commodity.rename(
            columns={
                "class": "课别",
                "code": "单品编码",
                "name": "单品名称",
                "sm_sort": "小分类编码",
                "md_sort": "中分类编码",
                "bg_sort": "大分类编码",
                "sm_sort_name": "小分类名称",
                "md_sort_name": "中分类名称",
                "bg_sort_name": "大分类名称",
            },
            inplace=True,
        )
        commodity.to_excel(f"{output_path}/commodity.xlsx", index=False)

        running = pd.read_csv(f"{input_path}running.csv")
        if not isinstance(running["code"].iloc[0], str):
            running["code"] = running["code"].astype("Int64").astype(str)
        # running按最终形成基准的commodity中的code进行第一次筛选
        running = running[running["code"].isin(commodity["单品编码"])]
        running["selldate"] = pd.to_datetime(running["selldate"])
        running["selldate"] = running["selldate"].apply(lambda x: x.date())
        # 将running中selldate和code，与account中日期和单品编码相同的筛选出来
        running = running.merge(
            account[["日期", "单品编码"]],
            how="inner",
            left_on=["selldate", "code"],
            right_on=["日期", "单品编码"],
        )
        # merge之后马上剔除多余的列，不能留到后面有相同列名的时候一起剔除，否则会剔除掉account中的日期和单品编码
        running.drop(columns=["日期", "单品编码"], inplace=True)
        running.sort_values(by=["selldate", "code"], inplace=True)
        running["打折销售"] = ["是" if x > 0 else "否" for x in running["sum_disc"]]
        assert (
            running["打折销售"].value_counts().values.sum() == running.shape[0]
        ), "流水表打折销售列计算有误"

        # 如果苹果在commodity的小分类名称中存在，需要输出running表，用于question_3_pre.py
        if "苹果" in commodity["小分类名称"].unique():
            running.to_csv(f"{output_path_self_use}/running.csv", index=False)

        running.rename(
            columns={
                "selldate": "销售日期",
                "selltime": "扫码销售时间",
                "class": "课别",
                "code": "单品编码",
                "amount": "销量",
                "price": "销售单价(元)",
                "type": "销售类型",
            },
            inplace=True,
        )
        running.drop(columns=["organ", "sum_disc", "sum_sell"], inplace=True)

        run_com = pd.merge(running, commodity, on=["课别", "单品编码"], how="left")
        print(run_com["小分类名称"].value_counts().sort_values(ascending=False), "\n")
        print(
            f"小分类编码与名称不唯一匹配的个数：{sum(run_com['小分类编码'].value_counts().sort_values(ascending=False).values != run_com['小分类名称'].value_counts().sort_values(ascending=False).values)}",
            "\n",
        )
        print(f"running.isnull().sum():\n{running.isnull().sum()}", "\n")
        print("running.info()", "\n", running.info(), "\n")

        try:
            running.to_excel(f"{output_path}/running.xlsx", index=False)
        except:
            running.to_csv(
                f"{output_path}/running.csv", index=False, encoding="utf-8-sig"
            )  # encoding='utf-8-sig'，解决excel打开，中文是乱码的问题
        print(running["销售类型"].value_counts().sort_values(ascending=False), "\n")
        print(running["打折销售"].value_counts().sort_values(ascending=False), "\n")

        # 单独计算7月1日个品类和单品的销量、进价、售价
        # 统计单品的日销量、进价、日平均售价
        last_day = pd.to_datetime(last_day).date()
        run_com = run_com[run_com["销售日期"] == last_day]
        run_com["销售金额"] = run_com["销量"] * run_com["销售单价(元)"]
        run_com = (
            run_com.groupby(
                ["销售日期", "单品编码", "单品名称", "小分类编码", "小分类名称"]
            )
            .agg({"销量": "sum", "销售金额": "sum"})
            .reset_index()
        )
        run_com["日平均售价"] = run_com["销售金额"] / run_com["销量"]
        run_com.dropna(how="any", inplace=True)

        sale_price_cost = run_com.merge(
            account,
            left_on=["销售日期", "单品编码"],
            right_on=["日期", "单品编码"],
            how="left",
        )
        sale_price_cost.drop(columns=["日期", "课别"], inplace=True)

        # 统计品类的日销量、日平均进价、日平均售价
        sale_price_cost["日成本"] = (
            sale_price_cost["销量"] * sale_price_cost["当天进货单价(元)"]
        )
        sale_price_cost_sm = (
            sale_price_cost.groupby(["销售日期", "小分类编码", "小分类名称"])
            .agg({"销量": "sum", "销售金额": "sum", "日成本": "sum"})
            .reset_index()
        )

        # 创建 ExcelWriter 对象
        writer = pd.ExcelWriter(f"{output_path_match}/sale_price_cost.xlsx")
        # 将 sale_price_cost 输出到 '单品' sheet
        sale_price_cost.to_excel(
            writer,
            sheet_name="单品",
            index=False,
            encoding="utf-8-sig",
            header=True,
            float_format="%.2f",
        )
        # 将 sale_price_cost_sm 输出到 '品类' sheet
        sale_price_cost_sm.to_excel(
            writer,
            sheet_name="品类",
            index=False,
            encoding="utf-8-sig",
            header=True,
            float_format="%.2f",
        )
        # 保存 Excel 文件
        writer.save()
