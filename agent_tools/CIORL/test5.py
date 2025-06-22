from datetime import datetime
# File: train_meta_agent_all.py

import os
import sys
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# === Add project root to path ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Observation Builder from CSVs ===
def precompute_observations_from_files(summary_df, rag_suggestions_path):
    rag_df = pd.read_csv(rag_suggestions_path)
    rag_df["date"] = pd.to_datetime(rag_df["date"]).dt.date
    rag_lookup = rag_df.set_index("date").to_dict(orient="index")

    extra_features = [
        "macd", "rsi_30", "cci_30", "vix", "turbulence", "close_30_sma", "close_60_sma"
    ]
    cached_obs = []

    for _, row in summary_df.iterrows():
        obs = []

        # 4 base agents: technical, sentiment, macro, risk
        for key in ["technical", "sentiment", "macro", "risk"]:
            action = row.get(f"{key}_action", "hold")
            confidence = float(row.get(f"{key}_confidence", 0.0))
            for act in ["buy", "hold", "sell"]:
                obs.append(confidence if act == action else 0.0)

        # rag agent (from rag_agent_suggestions.csv)
        date_key = pd.to_datetime(row["date"]).date()
        rag_result = rag_lookup.get(date_key, None)
        if rag_result:
            rag_action = rag_result.get("action", "hold")
            rag_conf = float(rag_result.get("confidence", 0.0))
        else:
            rag_action, rag_conf = "hold", 0.0

        for act in ["buy", "hold", "sell"]:
            obs.append(rag_conf if act == rag_action else 0.0)

        # holding flag
        obs.append(0.0)

        # extra features
        for feat in extra_features:
            obs.append(row.get(feat, 0.0))

        cached_obs.append(np.array(obs, dtype=np.float32))

    return cached_obs


# === MetaCIO Environment ===
class MetaCIOEnv(gym.Env):
    def __init__(self, data, cached_obs, trading_cost=0.001, pnl_alpha=0.1):
        super(MetaCIOEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.cached_obs = cached_obs
        self.current_step = 0

        self.holding = False
        self.entry_price = 0.0
        self.shares = 0

        self.trading_cost = trading_cost  # e.g. 0.1%
        self.pnl_alpha = pnl_alpha        # 浮动 pnl 激励因子

        obs_dim = len(cached_obs[0])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 = buy, 1 = hold, 2 = sell

    def _get_obs(self):
        if self.current_step >= len(self.cached_obs):
            return np.zeros_like(self.cached_obs[0])
        obs = self.cached_obs[self.current_step].copy()
        obs[15] = 1.0 if self.holding else 0.0  # overwrite holding flag
        return obs

    def reset(self):
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.shares = 0
        return self._get_obs()

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row["close"]

        reward = 0.0

        # --- 动作执行 ---
        if action == 0 and not self.holding:
            self.holding = True
            self.entry_price = price
            self.shares = 1
            reward -= self.trading_cost

        elif action == 2 and self.holding:
            pnl = (price - self.entry_price) / self.entry_price
            reward += pnl - self.trading_cost
            self.holding = False
            self.entry_price = 0.0
            self.shares = 0

        elif action == 1 and self.holding:
            # 浮动盈亏奖励（未平仓）
            unrealized_pnl = (price - self.entry_price) / self.entry_price
            reward += self.pnl_alpha * unrealized_pnl

        # --- 环境推进 ---
        self.current_step += 1
        done = self.current_step >= len(self.data)

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Holding: {self.holding}, Entry: {self.entry_price}")


# === Multi-Ticker Training Entry ===
if __name__ == "__main__":
    base_path = "../../datasets/processed"
    tickers = ["AAPL", "GOOG", "TSLA", "COIN", "NFLX", "NIO", "AMZN", "MSFT"]

    for ticker in tickers:
        print(f"\n=== Processing {ticker} ===")
        ticker_dir = os.path.join(base_path, ticker)
        data_path = os.path.join(ticker_dir, "financial_with_news_macro_summary.csv")
        rag_path = os.path.join(ticker_dir, "rag_agent_suggestions.csv")
        cache_path = os.path.join(ticker_dir, "cached_observations.csv")
        model_path = os.path.join(ticker_dir, f"meta_cio_rl_{ticker}.zip")

        if not os.path.exists(data_path) or not os.path.exists(rag_path):
            print(f"[WARN] Skipping {ticker}: Missing data or suggestions.")
            continue

        df = pd.read_csv(data_path)  
        df = df.groupby("date").first().reset_index()

        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 划分验证集（train/val）和测试集（test）
        train_start, train_end= datetime(2022, 1, 3).date(), datetime(2022, 10, 4).date()
        test_start, test_end= datetime(2022, 10, 5).date(), datetime(2023, 6, 10).date()

        df_train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
        df_test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        # === 训练阶段 ===
        print(f"[INFO] Training on {len(df_train)} rows for {ticker}")
        cached_obs_train = precompute_observations_from_files(df_train, rag_path)
        env_train = MetaCIOEnv(df_train, cached_obs_train)
        model = PPO("MlpPolicy", env_train, verbose=1, tensorboard_log="./tensorboard_logs/", device="cpu")
        model.learn(total_timesteps=50000)
        model.save(model_path)
        print(f"[✅ TRAINED] Model saved to {model_path}")

        # === 测试阶段 ===
        print(f"[INFO] Testing on {len(df_test)} rows for {ticker}")
        cached_obs_test = precompute_observations_from_files(df_test, rag_path)
        env_test = MetaCIOEnv(df_test, cached_obs_test)

        obs = env_test.reset()
        done = False

        portfolio_values = []
        initial_cash = 1.0
        cash = initial_cash
        shares = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            row = env_test.data.iloc[env_test.current_step]
            price = row["close"]

            # 资产组合价值计算逻辑
            if action == 0 and shares == 0:  # buy
                shares = cash / price
                cash = 0.0
            elif action == 2 and shares > 0:  # sell
                cash = shares * price
                shares = 0.0

            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)

            obs, _, done, _ = env_test.step(action)

        # === 评估指标计算 ===
        portfolio_values = np.array(portfolio_values)
        cr = (portfolio_values[-1] / portfolio_values[0] - 1) * 100  # Cumulative Return %
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sr = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0  # Sharpe Ratio
        cummax = np.maximum.accumulate(portfolio_values)
        mdd = ((portfolio_values - cummax) / cummax).min() * 100  # Max Drawdown %

        print(f"[📊 TEST RESULT] {ticker}")
        print(f"  ▸ CR (Cumulative Return): {cr:.2f}%")
        print(f"  ▸ SR (Sharpe Ratio): {sr:.4f}")
        print(f"  ▸ MDD (Max Drawdown): {mdd:.2f}%")