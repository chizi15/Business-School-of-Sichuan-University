import pandas as pd
import numpy as np
import datetime

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 8)


organ = 10
sort = 20
decim = 4

if organ == 1:
    organ = "DH"
    sort = "鸡蛋"
else:
    organ = "HLJ"
    if sort == 1:
        sort = "肉"  # bg_sort_name
    elif sort == 2:
        sort = "绿叶类"  # md_sort_name
    else:
        sort = "茄类"  # sm_sort_name

account = pd.read_csv(
    f"D:\Work info\WestUnion\data\origin\\{organ}\\account.csv",
    dtype={"code": str},
    parse_dates=["busdate"],
    infer_datetime_format=True,
)
print(
    f"\naccount:\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n"
    f"{account.isnull().sum()}\n\nisnull-rows-ratio-avg(%):"
    f"\n{round(sum(account.isnull().sum()) / (len(account) * max(1, sum(account.isnull().any()))) * 100, decim)}\n"
)
running = pd.read_csv(
    f"D:\Work info\WestUnion\data\origin\\{organ}\\running.csv",
    parse_dates=["selldate"],
    dtype={"code": str},
)
running["selltime"] = running["selltime"].apply(lambda x: x[:8])  # 截取出时分秒
running["selltime"] = pd.to_datetime(running["selltime"], format="%H:%M:%S")
running["selltime"] = running["selltime"].dt.time  # 去掉to_datetime自动生成的年月日
print(
    f"\nrunning:\n\nshape: {running.shape}\n\ndtypes:\n{running.dtypes}\n\nisnull-columns:\n"
    f"{running.isnull().sum()}\n\nisnull-rows-ratio-avg(%):"
    f"\n{round(sum(running.isnull().sum()) / (len(running) * max(1, sum(running.isnull().any()))) * 100, decim)}\n"
)
stock = pd.read_csv(
    f"D:\Work info\WestUnion\data\origin\\{organ}\\stock.csv",
    parse_dates=["busdate"],
    infer_datetime_format=True,
    dtype={"code": str},
)
print(
    f"\nstock:\n\nshape: {stock.shape}\n\ndtypes:\n{stock.dtypes}\n\nisnull-columns:\n"
    f"{stock.isnull().sum()}\n\nisnull-rows-ratio-avg(%):"
    f"\n{round(sum(stock.isnull().sum()) / (len(stock) * max(1, sum(stock.isnull().any()))) * 100, decim)}\n"
)

match organ:
    case "DH":
        commodity = pd.read_csv(
            f"D:\Work info\WestUnion\data\origin\\{organ}\\commodity.csv"
        )
        prediction = pd.read_csv(
            f"D:\Work info\WestUnion\data\origin\\{organ}\\prediction.csv",
            dtype={"code": str},
            parse_dates=["busdate"],
            infer_datetime_format=True,
        )
        # screening commodity sheet by requirement
        comodt = commodity[commodity["name"].str.contains(f"{sort}")]
        comodt_seg = comodt.drop(
            labels=commodity[commodity["name"].str.contains(f"{sort}面")].index
        )
        # screening other four sheets by requirement
        acct_seg = account[account["code"].isin(comodt_seg["code"])]
        run_seg = running[running["code"].isin(comodt_seg["code"])]
        stok_seg = stock[stock["code"].isin(comodt_seg["code"])]
        pred_seg = prediction[prediction["code"].isin(comodt_seg["code"])]

    case _:
        commodity = pd.read_csv(
            f"D:\Work info\WestUnion\data\origin\\{organ}\\commodity.csv",
            dtype={"code": str, "sm_sort": str, "md_sort": str, "bg_sort": str},
        )
        prediction = pd.read_csv(
            f"D:\Work info\WestUnion\data\origin\\{organ}\\fresh-forecast-order.csv",
            dtype={"bg_sort": str, "md_sort": str, "sm_sort": str, "code": str},
            names=[
                "Unnamed",
                "organ",
                "class",
                "bg_sort",
                "bg_sort_name",
                "md_sort",
                "md_sort_name",
                "sm_sort",
                "sm_sort_name",
                "code",
                "name",
                "busdate",
                "theory_sale",
                "real_sale",
                "predict",
                "advise_order",
                "real_order",
            ],
            header=0,
            parse_dates=["busdate"],
            infer_datetime_format=True,
        )
        promotion = pd.read_csv(
            f"D:\Work info\WestUnion\data\origin\\{organ}\\promotion.csv",
            parse_dates=["busdate"],
            infer_datetime_format=True,
            dtype={"code": str},
        )
        print(
            f"\npromotion:\n\nshape: {promotion.shape}\n\ndtypes:\n{promotion.dtypes}\n\nisnull-columns:\n"
            f"{promotion.isnull().sum()}\n\nisnull-rows-ratio-avg(%):"
            f"\n{round(sum(promotion.isnull().sum()) / (len(promotion) * max(1, sum(promotion.isnull().any()))) * 100, decim)}\n"
        )
        # screening commodity and promotion sheets and other four sheets by requirement
        match sort:
            case "肉":
                comodt_seg = commodity[
                    commodity["bg_sort_name"].str.contains(f"{sort}")
                ]
                critical_day = datetime.datetime(2021, 10, 31)
                abnormal_day = pd.DataFrame([critical_day])
                abnormal_day = abnormal_day[0]
            case "绿叶类":
                comodt_seg = commodity[
                    commodity["md_sort_name"].str.contains(f"{sort}")
                ]
                critical_day = datetime.datetime(2022, 2, 4)
                abnormal_day = pd.DataFrame([critical_day])
                abnormal_day = abnormal_day[0]
            case _:
                comodt_seg = commodity[
                    commodity["sm_sort_name"].str.contains(f"{sort}")
                ]
                comodt_seg = comodt_seg[
                    ~(
                        (comodt_seg["name"] == "贵妃西红柿")
                        | (comodt_seg["name"] == "岭上水果番茄")
                        | (comodt_seg["name"] == "草莓番茄")
                        | (comodt_seg["name"] == "奶油西红柿")
                        | (comodt_seg["name"] == "巧克力西红柿")
                    )
                ]
                # 剔除异常日期
                critical_day = datetime.datetime(2022, 2, 4)
                abnormal_day = pd.DataFrame(
                    [
                        datetime.datetime(2022, 8, 18)
                        + pd.to_timedelta(np.arange(9), unit="d")
                    ]
                )
                abnormal_day = pd.concat(
                    [abnormal_day, pd.DataFrame([datetime.datetime(2022, 10, 21)])],
                    axis=1,
                )
                abnormal_day = abnormal_day.T
                abnormal_day.reset_index(drop=True, inplace=True)
                # 用列名提取df的第一列，以满足下面“acct_seg = acct_seg[acct_seg['busdate'] != abnormal_day.loc[_]]”的要求
                abnormal_day = abnormal_day[0]

        # screening other five sheets by requirement
        prom_seg = promotion[promotion["code"].isin(comodt_seg["code"])]
        acct_seg = account[account["code"].isin(comodt_seg["code"])]
        acct_seg = acct_seg[acct_seg["busdate"] > critical_day]
        for _ in range(len(abnormal_day)):
            acct_seg = acct_seg[acct_seg["busdate"] != abnormal_day.loc[_]]
        run_seg = running[running["code"].isin(comodt_seg["code"])]
        run_seg = run_seg[run_seg["selldate"] > critical_day]
        for _ in range(len(abnormal_day)):
            run_seg = run_seg[run_seg["selldate"] != abnormal_day.loc[_]]
        stok_seg = stock[stock["code"].isin(comodt_seg["code"])]
        stok_seg = stok_seg[stok_seg["busdate"] > critical_day]
        for _ in range(len(abnormal_day)):
            stok_seg = stok_seg[stok_seg["busdate"] != abnormal_day.loc[_]]
        pred_seg = prediction[prediction["code"].isin(comodt_seg["code"])]
        pred_seg = pred_seg[pred_seg["busdate"] > critical_day]
        for _ in range(len(abnormal_day)):
            pred_seg = pred_seg[pred_seg["busdate"] != abnormal_day.loc[_]]

print(
    f"\ncommodity:\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n"
    f"{commodity.isnull().sum()}\n\nisnull-rows-ratio-avg(%):"
    f"\n{round(sum(commodity.isnull().sum()) / (len(commodity) * max(1, sum(commodity.isnull().any()))) * 100, decim)}\n"
)
print(
    f"\nprediction:\n\nshape: {prediction.shape}\n\ndtypes:\n{prediction.dtypes}\n\nisnull-columns:\n"
    f"{prediction.isnull().sum()}\n\nisnull-rows-ratio-avg(%):"
    f"\n{round(sum(prediction.isnull().sum()) / (len(prediction) * max(1, sum(prediction.isnull().any()))) * 100, decim)}\n"
)

# merge sheets in order
match organ:
    case "DH":
        com_acct_seg = pd.merge(comodt_seg, acct_seg, how="left", on=["code"])
        com_acct_stk_seg = pd.merge(
            com_acct_seg, stok_seg, how="left", on=["organ", "code", "busdate"]
        )
        com_acct_stk_seg.drop(columns=["name"], inplace=True)
        com_acct_stk_pred_seg = pd.merge(
            com_acct_stk_seg, pred_seg, how="outer", on=["organ", "code", "busdate"]
        )
    case _:
        com_acct_seg = pd.merge(comodt_seg, acct_seg, how="left", on=["class", "code"])
        com_acct_stk_seg = pd.merge(
            com_acct_seg, stok_seg, how="left", on=["organ", "class", "code", "busdate"]
        )
        com_acct_stk_seg.drop(
            columns=["name", "sm_sort_name", "md_sort_name", "bg_sort_name"],
            inplace=True,
        )
        pred_seg = pred_seg.drop(
            columns=["Unnamed", "bg_sort_name", "md_sort_name", "sm_sort_name", "name"]
        )
        com_acct_stk_pred_seg = pd.merge(
            com_acct_stk_seg,
            pred_seg,
            how="outer",
            on=["organ", "class", "bg_sort", "md_sort", "sm_sort", "code", "busdate"],
        )
        com_acct_stk_pred_seg = pd.merge(
            com_acct_stk_pred_seg, prom_seg, how="left", on=["organ", "code", "busdate"]
        )

# derive merged sheet and running sheet
com_acct_stk_pred_seg.to_csv(
    f"D:\\Work info\\WestUnion\\data\\processed\\{organ}\\com_acct_stk_pred_{sort}.csv",
    encoding="utf_8_sig",
    index=False,
)
run_seg.to_csv(
    f"D:\\Work info\\WestUnion\\data\\processed\\{organ}\\run_{sort}.csv",
    encoding="utf_8_sig",
    index=False,
)
