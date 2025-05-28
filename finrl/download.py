import os
import pandas as pd
import yfinance as yf
from typing import List, Optional


class YahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: List[str], save_path: str = "./dataset"):
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
            file_path = os.path.join(self.save_path, f"{'_'.join(self.ticker_list)}.csv")
            data_df.to_csv(file_path, index=False)
            print(f"📁 数据已保存至 {file_path}")

        return data_df
