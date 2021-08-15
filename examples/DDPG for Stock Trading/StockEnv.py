import random

import numpy as np

import gym
from gym import spaces

# 默认的一些数据，用于归一化属性值
MAX_ACCOUNT_BALANCE = 214748        # 组大的账户财产
MAX_NUM_SHARES = 214748             # 最大的手数
MAX_SHARE_PRICE = 5000              # 最大的单手价格
MAX_VOLUME = 1000e6                 # 最大的成交量
MAX_AMOUNT = 3e5                    # 最大的成交额
MAX_OPEN_POSITIONS = 5              # 最大的持仓头寸
MAX_STEPS = 500                     # 最大的交互次数
MAX_DAY_CHANGE = 1                  # 最大的日期改变
max_loss =-50000                    # 最大的损失
max_predict_rate = 4                # 最大的预测率
INITIAL_ACCOUNT_BALANCE = 10000     # 初始的金钱


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # 动作的可能情况：买入x%, 卖出x%, 观望
        self.action_space = spaces.Box(
            low=np.array([-3, 0]), high=np.array([3, 1]), dtype=np.float32)

        # 环境状态的维度
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float32)

    
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    
    # 处理状态
    def _next_observation(self):
        # 有些股票数据缺失一些数据，处理一下
        d10 = self.df.loc[self.current_step, 'peTTM'] / 1e4
        d11 = self.df.loc[self.current_step, 'pbMRQ'] / 100
        d12 = self.df.loc[self.current_step, 'psTTM'] / 100
        if np.isnan(d10):       # 某些数据是0.00000000e+00，如果是nan会报错
            d10 = d11 = d12 = 0.00000000e+00
        obs = np.array([
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,
            self.df.loc[self.current_step, 'adjustflag'] / 10,
            self.df.loc[self.current_step, 'tradestatus'] / 1,
            self.df.loc[self.current_step, 'pctChg'] / 100,
            d10,
            d11,
            d12,
            self.df.loc[self.current_step, 'pctChg'] / 1e3,
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        return obs


    # 执行当前动作，并计算出当前的数据（如：资产等）
    def _take_action(self, action):
        # 随机设置当前的价格，其范围上界为当前时间点的价格
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])
        action_type = action[0]
        amount = action[1]
        if action_type > 1:     # 买入amount%
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < -1:  # 卖出amount%
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        # 计算出执行动作后的资产净值
        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


    # 与环境交互
    def step(self, action):
        # 在环境内执行动作
        self._take_action(action)
        done = False

        # 判断是否终止
        self.current_step += 1
        if self.max_net_worth >= INITIAL_ACCOUNT_BALANCE * max_predict_rate:
            done = True
        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0  # loop training

            done = True

        delay_modifier = (self.current_step / MAX_STEPS)

        # 计算相对收益比，并据此来计算奖励
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        profit_percent = profit / INITIAL_ACCOUNT_BALANCE
        if profit_percent>=0:
            reward = max(1,profit_percent/0.001)
        else:
            reward = -100

        if self.net_worth <= 0 :
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}


    # 重置环境
    def reset(self, new_df=None):
        # 重置环境的变量为初始值
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.count = 0
        self.interval = 5

        # 传入环境数据集
        if new_df:
            self.df = new_df

        self.current_step = 0

        return self._next_observation()


    # 显示环境至屏幕
    def render(self, mode='human'):
        # 打印环境信息
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-'*30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        return profit