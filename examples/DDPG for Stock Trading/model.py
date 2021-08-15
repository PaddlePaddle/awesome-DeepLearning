import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import copy

# 是否使用GPU
device = paddle.get_device()


# 动作网络：输出连续的动作信号
class Actor(nn.Layer):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # 输出层激活函数采用tanh，将输出映射至[-1,1]
        return F.tanh(self.l3(a))


# 值函数网络：评价一个动作的价值
class Critic(nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(paddle.concat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


# DDPG算法模型    
class DDPGModel(object):
    def __init__(self, state_dim, action_dim, max_action, gamma = 0.99, tau = 0.001):
        # 动作网络与目标动作网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(parameters=self.actor.parameters(), learning_rate=1e-4)

        # 值函数网络与目标值函数网络
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(parameters=self.critic.parameters(), weight_decay=1e-2)

        self.gamma = gamma
        self.tau = tau


    # 根据当前状态，选择动作：过一个动作网络得到动作
    def select_action(self, state):
        state = paddle.to_tensor(state.reshape(1, -1), dtype='float32', place=device)
        return self.actor(state).numpy().flatten()

    
    # 训练函数
    def train(self, replay_buffer, batch=64):
        # 从缓存容器中采样
        state, action, next_state, reward, done = replay_buffer.sample(batch)

        # 计算目标网络q值
        q_target = self.critic_target(next_state, self.actor_target(next_state))
        q_target = reward + ((1- done) * self.gamma * q_target).detach()

        # 计算当前网络q值
        q_eval = self.critic(state, action)

        # 计算值网络的损失函数
        critic_loss = F.mse_loss(q_eval, q_target)
        # print(critic_loss)

        # 梯度回传，优化网络参数
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算动作网络的损失函数
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # print(actor_loss)

        # 梯度回传，优化网络参数
        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络参数
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.set_value(target_param * (1.0 - self.tau) + param * self.tau)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.set_value(target_param * (1.0 - self.tau) + param * self.tau)
        

    # 保存模型参数    
    def save(self, filename):
        paddle.save(self.critic.state_dict(), filename + '_critic')
        paddle.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')

        paddle.save(self.actor.state_dict(), filename + '_actor')
        paddle.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
        

    # 导入模型参数
    def load(self, filename):
        self.critic.set_state_dict(paddle.load(filename + '_critic'))
        self.critic_optimizer.set_state_dict(paddle.load(filename + '_critic_optimizer'))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.set_state_dict(paddle.load(filename + '_actor'))
        self.actor_optimizer.set_state_dict(paddle.load(filename + '_actor_optimizer'))
        self.actor_target = copy.deepcopy(self.actor)

        