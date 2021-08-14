import random

import numpy as np

import gym
from gym import spaces

MAX_ACCOUNT_BALANCE = 214748
MAX_NUM_SHARES = 214748
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e6
MAX_AMOUNT = 3e5
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 500
MAX_DAY_CHANGE = 1
max_loss =-50000
max_predict_rate = 4
INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([-3, 0]), high=np.array([3, 1]), dtype=np.float32)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float32)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _next_observation(self):
        # 有些股票数据缺失一些数据，这里处理一下，不处理报错
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

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])
        action_type = action[0]
        amount = action[1]
        if action_type > 1:     #1
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < -1:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        done = False

        self.current_step += 1
        if self.max_net_worth >= INITIAL_ACCOUNT_BALANCE * max_predict_rate:
            done = True
        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0  # loop training

            done = True

        delay_modifier = (self.current_step / MAX_STEPS)

        # profits
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

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.count = 0
        self.interval = 5

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-'*30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        return profit