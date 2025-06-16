import os
import pandas as pd
import yfinance as yf
from typing import List, Optional


class YahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: List[str], save_path: str = "../../datasets/meta"):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def fetch_data(self, proxy: Optional[str] = None, save_csv: bool = True) -> pd.DataFrame:
        data_df = pd.DataFrame()

        for tic in self.ticker_list:
            print(f"正在下载 {tic} 的数据...")
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date, proxy=proxy)
            if temp_df.empty:
                print(f"⚠️ 没有获取到 {tic} 的数据。")
                continue

            temp_df.columns = temp_df.columns.get_level_values(0)
            print(temp_df.columns)
            temp_df["tic"] = tic
            temp_df = temp_df.reset_index()

            # 标准化列
            new_df = pd.DataFrame({
                'date': temp_df['Date'],
                'open': temp_df['Open'],
                'high': temp_df['High'],
                'low': temp_df['Low'],
                'close': temp_df['Close'],  # 用 Close 表示最终价
                'volume': temp_df['Volume'],
                'tic': temp_df['tic'],
            })


            new_df["day"] = new_df["date"].dt.dayofweek
            new_df["date"] = new_df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))

            data_df = pd.concat([data_df, new_df], ignore_index=True)

        # 清洗
        data_df = data_df.dropna()
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        print("✅ 下载完成，数据维度：", data_df.shape)

        if save_csv:
            # 构造路径
            folder_path = os.path.join(self.save_path, f"{self.start_date}")  # 比如 '../../datasets/meta/2024-08-01_'
            os.makedirs(folder_path, exist_ok=True)  # 自动创建多级目录（如果已存在不会报错）

            file_path = os.path.join(folder_path, f"{'_'.join(self.ticker_list)}.csv")
            data_df.to_csv(file_path, index=False)
            print(f"📁 数据已保存至 {file_path}")

        return data_df


if __name__ == "__main__":
    downloader = YahooDownloader(
        start_date="2020-01-01",
        end_date="2025-04-01",
        ticker_list = [
    # 科技
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "INTC",# 金融
    "JPM", "BAC", "GS",# 能源
    "XOM", "CVX",# 医疗
    "JNJ", "PFE", "UNH",# 消费
    "PG", "KO", "DIS", "WMT", "NIO","NFLX","COIN"]
    )

    df = downloader.fetch_data(proxy="http://127.0.0.1:7890")  # 如果你开了代理

    print(df.head())


# filename: download_data.py
# import yfinance as yf
# import os

# def download_stock_data(tickers, start_date="2015-01-01", end_date="2024-12-31", save_dir="dataset"):
#     os.makedirs(save_dir, exist_ok=True)

#     for ticker in tickers:
#         print(f"正在下载 {ticker} 的数据...")
#         proxy = "http://127.0.0.1:7890"  # 或 socks5://127.0.0.1:1080
#         data = yf.download(ticker, start=start_date, end=end_date, proxy=proxy)
#         file_path = os.path.join(save_dir, f"3.csv")
#         data.to_csv(file_path)
#         print(f"{ticker} 的数据已保存到 {file_path}\n")

# if __name__ == "__main__":
#     stock_list = ["AAPL"]  # 可根据需要改成你关注的股票
#     download_stock_data(stock_list)
