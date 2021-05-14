# 马尔科夫决策过程

<center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/reinforcement_learning/pacman.jpg?raw=true" alt="pacman" style="zoom:80%;" /></center>

<center>
  图1: 经典吃豆人游戏
  <br></br>
</center>

在经典的吃豆人游戏中，吃豆人通过对环境进行观察，选择上下左右四种动作中的一种进行自身移动，吃掉豆子获得分数奖励，并同时躲避幽灵防止被吃。吃豆人每更新一次动作后，都会获得环境反馈的新的状态，以及对该动作的分数奖励。在这个学习过程中，吃豆人就是智能体，游戏地图、豆子和幽灵位置等即为环境，而智能体与环境交互进行学习最终实现目标的过程就是马尔科夫决策过程（Markov decision process，MDP）。

<center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/reinforcement_learning/MDP.png?raw=true" alt="MDP" style="zoom:60%" /></center>

<center>
  图2: 马尔科夫决策过程中的智能体-环境交互
  <br></br>
</center>

上图形式化的描述了强化学习的框架，智能体（Agent）与环境（Environment）交互的过程：在 $t$ 时刻，智能体在当前状态 $S_t$ 采取动作 $A_t$。在下一时刻 $t+1$，智能体接收到环境反馈的对于动作 $A_t$ 的奖励 $R_{t+1}$，以及该时刻状态 $S_{t+1}$。从而，MDP和智能体共同给出一个轨迹：

$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,S_3,A_3,...
$$
接下来，更具体地定义以下标识：

* $S_t$ 是有限的状态集合
* $A_t$ 是有限的动作集合
* $P$ 是基于环境的状态转移矩阵：其中每一项为智能体在某个状态 $s$ 下，采取动作 $a$ 后，与环境交互后转移到其他状态 $s^{'}$ 的概率值，表示为 $P(S_{t+1}=s^{'}|s_{t}=s,a_{t}=a)$
* R是奖励函数：智能体在某个状态 $s$ 下，采取动作 $a$ 后，与环境交互后所获得的奖励，表示为 $R(s_{t}=s,a_{t}=a)$
* $\gamma$ 是折扣因子(discounted factor)，取值区间为 $[0,1]$

所以MDP过程可以表示为 $(S,A,P,R,\gamma)$，如果该过程中的状态转移矩阵 $P$ 和奖励 $R(s,a)$ 对智能体都是可见的，我们称这样的Agent为Model-based Agent，否则称为Model-free Agent。