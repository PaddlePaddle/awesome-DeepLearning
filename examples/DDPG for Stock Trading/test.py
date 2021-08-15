import numpy as np
import paddle
import argparse
import os
from visualdl import LogWriter


import model
import ReplayBuffer
import StockEnv
import pandas as pd


# 导入数据
df = pd.read_csv('data/data102715/test.csv')
# df = df.sort_values('date')
writer = LogWriter('./log/test')

# 测试环境使用的随机种子
eval_seed = [53, 47, 99, 107, 1, 17, 57, 97, 179, 777]

# 评估模型的函数
def eval_policy(policy, df, seed, eval_episodes=10):
    avg_reward = 0.
    for epi in range(eval_episodes):
        # 初始化评估环境并设定随机种子
        eval_env = StockEnv.StockTradingEnv(df)
        eval_env.seed(seed + eval_seed[epi])
        
        # 初始化评估环境
        state, done = eval_env.reset(), False
        t = 0
        epi_reward = 0

        # 模型与环境交互
        while not done:
            action = policy.select_action(state)
            action[0] *=3
            state, reward, done, _ = eval_env.step(action)
            writer.add_scalar(tag='reward', step=t, value=reward)
            t += 1
            epi_reward += reward
            avg_reward += reward
        
        # 可视化整个幕的奖励
        writer.add_scalar(tag='episode_reward', step=epi, value=epi_reward)
    
    # 计算得到平均奖励
    avg_reward /= eval_episodes

    print('-----------------------------------------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward:.3f}')
    print('-----------------------------------------')

    return avg_reward

# 默认的超参数
default_seed = 123

# 参数语法分析器
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=default_seed, type=int)

# 参数
args = parser.parse_args()

file_name = f'DDPG_Stock_{args.seed}'

if __name__ == '__main__':

    # 初始化一个例子环境，用于得到环境的属性信息，例如：状态和动作的维度
    env = StockEnv.StockTradingEnv(df)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[1])
    print(state_dim, action_dim, max_action)

    kwarg = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        "max_action": max_action,
    }

    # 创建模型：DDPG策略
    policy = model.DDPGModel(**kwarg)

    # 导入训练完成的模型参数
    policy_file = file_name
    policy.load(f"./models/{policy_file}")

    # 做评估
    evaluations = [eval_policy(policy, df, args.seed)]

    