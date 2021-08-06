import numpy as np
import paddle
import argparse
import os
from visualdl import LogWriter


import model
import ReplayBuffer
import StockEnv
import pandas as pd


# get data
df = pd.read_csv('data/data102715/test.csv')
# df = df.sort_values('date')
writer = LogWriter('./log/test')
eval_seed = [53, 47, 99, 107, 1, 17, 57, 97, 179, 777]

def eval_policy(policy, df, seed, eval_episodes=10):
    avg_reward = 0.
    for epi in range(eval_episodes):
        eval_env = StockEnv.StockTradingEnv(df)
        eval_env.seed(seed + eval_seed[epi])
        # TODO: reset env
        state, done = eval_env.reset(), False
        t = 0
        epi_reward = 0
        while not done:
            action = policy.select_action(state)
            # TODO: step with env
            action[0] *=3
            state, reward, done, _ = eval_env.step(action)
            writer.add_scalar(tag='reward', step=t, value=reward)
            t += 1
            epi_reward += reward
            avg_reward += reward
        writer.add_scalar(tag='episode_reward', step=epi, value=epi_reward)
    avg_reward /= eval_episodes

    print('-----------------------------------------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward:.3f}')
    print('-----------------------------------------')

    return avg_reward

# default hyperparams
default_seed = 123

# args
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=default_seed, type=int)

args = parser.parse_args()

file_name = f'DDPG_Stock_{args.seed}'

if __name__ == '__main__':

    env = StockEnv.StockTradingEnv(df)
    # TODO: set valeus according to env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[1])
    print(state_dim, action_dim, max_action)

    kwarg = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        "max_action": max_action,
    }

    # set model policy
    policy = model.DDPGModel(**kwarg)

    policy_file = file_name
    policy.load(f"./models/{policy_file}")

    evaluations = [eval_policy(policy, df, args.seed)]


