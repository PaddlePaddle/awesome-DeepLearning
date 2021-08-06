import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import copy

device = paddle.get_device()

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
        # return self.max_action * F.tanh(self.l3(a))
        return F.tanh(self.l3(a))


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

    
class DDPGModel(object):
    def __init__(self, state_dim, action_dim, max_action, gamma = 0.99, tau = 0.001):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(parameters=self.actor.parameters(), learning_rate=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(parameters=self.critic.parameters(), weight_decay=1e-2)

        self.gamma = gamma
        self.tau = tau

    
    def select_action(self, state):
        state = paddle.to_tensor(state.reshape(1, -1), dtype='float32', place=device)
        return self.actor(state).numpy().flatten()

    
    def train(self, replay_buffer, batch=64):
        # sample
        state, action, next_state, reward, done = replay_buffer.sample(batch)

        # compute q target
        q_target = self.critic_target(next_state, self.actor_target(next_state))
        q_target = reward + ((1- done) * self.gamma * q_target).detach()

        # get q eval
        q_eval = self.critic(state, action)

        # compute critic loss
        critic_loss = F.mse_loss(q_eval, q_target)
        # print(critic_loss)

        # optimize the critic
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # print(actor_loss)

        # optimize the actor
        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the froze target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.set_value(target_param * (1.0 - self.tau) + param * self.tau)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.set_value(target_param * (1.0 - self.tau) + param * self.tau)
        
        
    def save(self, filename):
        paddle.save(self.critic.state_dict(), filename + '_critic')
        paddle.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')

        paddle.save(self.actor.state_dict(), filename + '_actor')
        paddle.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
        

    def load(self, filename):
        self.critic.set_state_dict(paddle.load(filename + '_critic'))
        self.critic_optimizer.set_state_dict(paddle.load(filename + '_critic_optimizer'))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.set_state_dict(paddle.load(filename + '_actor'))
        self.actor_optimizer.set_state_dict(paddle.load(filename + '_actor_optimizer'))
        self.actor_target = copy.deepcopy(self.actor)

        
        
        
        
        
        
        
        
        
        



