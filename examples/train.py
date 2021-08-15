import numpy as np
import paddle
import argparse
import os
from visualdl import LogWriter


import model
import ReplayBuffer
import StockEnv
import pandas as pd


# 获得数据
df = pd.read_csv('data/data102715/train.csv')
# df = df.sort_values('date')


# 评估模型
def eval_policy(policy, df, seed, eval_episodes=10):
    # 创建评估环境，并设置随机种子
    eval_env = StockEnv.StockTradingEnv(df)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        # 初始化环境
        state, done = eval_env.reset(), False
        
        # 与环境交互
        while not done:
            action = policy.select_action(state)
            # TODO: step with env
            action[0] *= 3
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    
    # 计算平均奖励
    avg_reward /= eval_episodes

    print('-----------------------------------------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward:.3f}')
    print('-----------------------------------------')

    return avg_reward


# 默认的超参数
default_seed = 123          # 随机种子
default_batch = 64          # 批量大小
default_gamma = 0.95        # 折扣因子
default_tau = 0.005         # 当前网络参数比例，用于更新目标网络
default_timesteps = 2e5     # 训练步数
default_expl_noise = 0.1    # 高斯噪声
default_eval_freq = 6e3     # 评估模型的频率

# 参数语法解析器
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=default_seed, type=int)
parser.add_argument("--batch_size", default=default_batch, type=int)
parser.add_argument("--gamma", default=default_gamma)
parser.add_argument("--tau", default=default_tau)
parser.add_argument("--expl_noise", default=default_expl_noise)
parser.add_argument("--eval_freq", default=default_eval_freq, type=int)
parser.add_argument("--timesteps", default=default_timesteps, type=int)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--load_model", default="")
args = parser.parse_args()

file_name = f'DDPG_Stock_{args.seed}'
writer = LogWriter('./log/train')

if __name__ == '__main__':
    # 路径设置
    if not os.path.exists("./results"):
	    os.makedirs('./results')

    if args.save_model and not os.path.exists("./models"):
        os.makedirs('./models')

    # 根据数据集设置环境
    env = StockEnv.StockTradingEnv(df)

    # 设置随机种子
    env.seed(args.seed)
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    # T得到环境的参数信息（如：状态和动作的维度）
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[1])
    print(state_dim, action_dim, max_action)

    kwarg = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        "max_action": max_action,
        'gamma': args.gamma,
        'tau': args.tau
    }

    # 设置模型：DDPG算法
    policy = model.DDPGModel(**kwarg)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f'./models/{policy_file}')
    
    # 设置缓存容器
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim)

    # 评估初始环境：对照
    evaluations = [eval_policy(policy, df, args.seed)]    

    # 初始化环境
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # 与环境交互
    for t in range(int(args.timesteps)):

        episode_timesteps += 1

        # 根据状态得到动作
        action = (
            policy.select_action(np.array(state))
            + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
        ).clip(-max_action, max_action)
        action[0] *= 3
        print('action', action)

        # 在环境中执行动作
        next_state, reward, done, _ = env.step(action)
        print('reward', reward)
        writer.add_scalar(tag='reward', step=t, value=reward)

        # 将交互数据存入容器
        replay_buffer.add(state, action, next_state, reward, done)

        # 状态更新
        state = next_state
        episode_reward += reward

        # 算法训练
        policy.train(replay_buffer, args.batch_size)

        # 该轮交互结束
        if done:
            # 打印信息，重置状态
            print(f'Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}')
            # Reset environment
            writer.add_scalar(tag='episode_reward', step=episode_num, value= episode_reward)
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # 评估算法表现，并存储模型
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, df, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}") 

        