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
df = pd.read_csv('data/data102715/train.csv')
# df = df.sort_values('date')

def eval_policy(policy, df, seed, eval_episodes=10):
    # TODO: get env
    eval_env = StockEnv.StockTradingEnv(df)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        # TODO: reset env
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(state)
            # TODO: step with env
            action[0] *= 3
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    
    avg_reward /= eval_episodes

    print('-----------------------------------------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward:.3f}')
    print('-----------------------------------------')

    return avg_reward


# default hyperparams
default_seed = 123
default_batch = 64
default_gamma = 0.95
default_tau = 0.005
default_timesteps = 2e5
default_expl_noise = 0.1
default_eval_freq = 6e3

# args
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
    if not os.path.exists("./results"):
	    os.makedirs('./results')

    if args.save_model and not os.path.exists("./models"):
        os.makedirs('./models')

    # set env
    env = StockEnv.StockTradingEnv(df)

    # set seed
    env.seed(args.seed)
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    # TODO: set valeus according to env
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

    # set model policy
    policy = model.DDPGModel(**kwarg)

    if args.load_model != "":
        policy_file = filename if args.load_model == "default" else args.load_model
        policy.load(f'./models/{policy_file}')
    
    # set replay buffer
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim)

    # evaluate the untrained policy TODO: env.seed
    evaluations = [eval_policy(policy, df, args.seed)]    

    # TODO: env values
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.timesteps)):

        episode_timesteps += 1

        # get action from state
        action = (
            policy.select_action(np.array(state))
            + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
        ).clip(-max_action, max_action)
        action[0] *= 3
        print('action', action)

        # perform action
        next_state, reward, done, _ = env.step(action)
        print('reward', reward)
        writer.add_scalar(tag='reward', step=t, value=reward)

        # store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_reward += reward

        # train
        policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f'Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}')
            # Reset environment
            writer.add_scalar(tag='episode_reward', step=episode_num, value= episode_reward)
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, df, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}") 

        