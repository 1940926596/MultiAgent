import pandas as pd
import numpy as np
from download_data import YahooDownloader
from stockstats import StockDataFrame as Sdf
from datetime import datetime
import itertools
import config



class FeatureEngineer:
    def __init__(self, indicators=None, use_turbulence=False):
        self.indicators = indicators if indicators else config.INDICATORS
        self.use_turbulence = use_turbulence

    def preprocess(self, df):
        df = self._clean(df)
        df = self._add_indicators(df)
        df = self._add_vix(df)
        print("--------------------")
        if self.use_turbulence:
            df = self._add_turbulence(df)
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def _clean(self, df):
        df = df.sort_values(["date", "tic"])
        df = df[df["close"].notna()]
        return df

    def _add_indicators(self, df):
        df = df.sort_values(["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.indicators:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_df = pd.DataFrame(temp)
                    temp_df["tic"] = unique_ticker[i]
                    temp_df["date"] = df[df.tic == unique_ticker[i]]["date"].values
                    indicator_df = pd.concat([indicator_df, temp_df], ignore_index=True)
                except Exception as e:
                    print(f"Error processing indicator {indicator} for {unique_ticker[i]}: {e}")
            df = df.merge(indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left")

        df = df.sort_values(["date", "tic"])
        return df

    def _add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        ).fetch_data(proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        print(df.head())
        return df


    def _add_turbulence(self, df):
        df = df.copy()
        print(df.columns)
        df_pivot = df.pivot(index="date", columns="tic", values="close").pct_change()
        turbulence = []
        start = 252
        dates = df_pivot.index
        for i in range(len(dates)):
            if i < start:
                turbulence.append(0)
                continue
            hist = df_pivot.iloc[i - 252:i].dropna(axis=1)
            current = df_pivot.iloc[i][hist.columns]
            cov = hist.cov()
            diff = current - hist.mean()
            temp = diff.values @ np.linalg.pinv(cov.values) @ diff.values.T
            turbulence.append(temp if temp > 0 else 0)
        turbulence_df = pd.DataFrame({"date": dates, "turbulence": turbulence})
        df = df.merge(turbulence_df, on="date", how="left")
        return df


if __name__ == "__main__":
 
    # 读取原始数据
    df = pd.read_csv("../../datasets/meta/2022-01-01/AAPL_MSFT_GOOGL.csv")
    # print("\n索引：", df.index)

    # 添加技术指标和Turbulence
    fe = FeatureEngineer(use_turbulence=True)
    processed = fe.preprocess(df)

    # 补全缺失时间-股票组合，按日期和股票排序
    tickers = processed["tic"].unique().tolist()
    dates = pd.date_range(processed["date"].min(), processed["date"].max()).astype(str).tolist()
    combos = pd.DataFrame(itertools.product(dates, tickers), columns=["date", "tic"])
    processed_full = combos.merge(processed, on=["date", "tic"], how="left")

    # 保留有效日期并排序填充
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"]).fillna(0)

    # 确保 processed_full 的日期列是 datetime 类型
    processed_full['date'] = pd.to_datetime(processed_full['date'])

    # 使用标准日期格式筛选
    processed_full = processed_full[
        (processed_full['date'] >= '2024-08-01') & (processed_full['date'] <= '2025-04-01')
    ]

    file_path = "../../datasets/meta/2022-01-01/AAPL_MSFT_GOOGL_processors.csv"
    processed_full.to_csv(file_path, index=False)
    print(f"📁 数据已保存至 {file_path}")
    

# import pandas as pd

# df = pd.DataFrame({
# 'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
# 'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
# 'baz': [1, 2, 3, 4, 5, 6],
# 'zoo': ['x', 'y', 'z', 'q', 'w', 't']
# })

# df.pivot(index='foo', columns='bar', values='baz')
# print("\n维度：", df.shape)
# print("\n列名：", df.columns)
# print("\n索引：", df.index)