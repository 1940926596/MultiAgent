"""Contains methods and classes to collect data from
Yahoo Finance API
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic, start=self.start_date, end=self.end_date, proxy=proxy
            )

            print("结构信息：")
            temp_df.info()

            print("\n维度：", temp_df.shape)

            print("\n列名：", temp_df.columns)

            print("\n索引：", temp_df.index)

            print("\n前几行数据：")
            print(temp_df.head())
            

            # print(temp_df.columns)
            temp_df.columns = temp_df.columns.get_level_values(0)  # 只保留第一层（）
            temp_df["tic"] = tic
            # temp_df = temp_df.reset_index()  # 展平索引

            # print(temp_df.columns)

            # temp_df = temp_df.drop(labels="Price", axis=1)
            # print("111111111")
            # print(temp_df.head())


            temp_df = temp_df.reset_index()
            new_df = pd.DataFrame({'Date': temp_df['Date'],
                       'Open': temp_df['Open'],
                       'High': temp_df['High'],
                       'Low': temp_df['Low'],
                       'Close': temp_df['Close'],
                       'Adjcp': '',
                       'Volume': temp_df['Volume'],
                       'tic': temp_df['tic'],
                       })
            # print("0000000000000000000000000")
            # print(new_df.head())
            # print(len(new_df.columns))
            # print("0000000000000000000000000")
            # data_df = data_df.append(temp_df)
            # ------------------------------------------------------
            data_df = pd.concat([data_df, new_df])

        # print(data_df.head())
        # data_df.to_csv("~/data/data.csv", index=False)

        # reset the index, we want to use numbers as index instead of dates
        # data_df = data_df.reset_index()

        # print("2222222222")
        # print(temp_df.columns)
        # print(data_df.head())
        # print(len(data_df.columns))  # 看一下有多少列
        # print(data_df["Date"])

        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "adjcp",
                "close",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")

        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek

        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))

        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)

        # print("Display DataFrame: ", data_df.head())
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
