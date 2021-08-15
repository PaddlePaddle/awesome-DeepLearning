import numpy as np
import paddle


# 缓存容器：内容为{obs, act, obs_, reward, done}五元组
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e4)):
        self.max_size = max_size
        self.cur = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.next_states = np.zeros((max_size, state_dim))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))

        self.device = paddle.get_device()

    
    # 存入数据
    def add(self, state, action, next_state, reward, done):
        self.states[self.cur] = state
        self.actions[self.cur] = action
        self.next_states[self.cur] = next_state
        self.rewards[self.cur] = reward
        self.dones[self.cur] = done

        # 指针移动
        self.cur = (self.cur + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    
    # 采样
    def sample(self, batch):
        ids = np.random.randint(0, self.size, size=batch)

        # 返回paddle张量
        return (
            paddle.to_tensor(self.states[ids], dtype='float32', place=self.device),
            paddle.to_tensor(self.actions[ids], dtype='float32', place=self.device),
            paddle.to_tensor(self.next_states[ids], dtype='float32', place=self.device),
            paddle.to_tensor(self.rewards[ids], dtype='float32', place=self.device),
            paddle.to_tensor(self.dones[ids], dtype='float32', place=self.device)
        )

    