# 策略梯度定理

策略函数的参数化可表示为$\pi_{\theta}(s,a)$，其中$\theta$为一组参数，函数取值表示在状态$s$下选择动作$a$的概率。为了优化策略函数，首先需要有一个对策略函数优劣进行衡量的标准。假设强化学习问题的初始状态为$s_{0}$，则希望达到最大化的目标为


$$
J(\theta) := V_{\pi_{\theta}}(s_{0})
$$
其中，$v_{\pi_{\theta}}$是在策略$\pi_{\theta}$下的真实价值函数，这个价值函数代表衡量一个状态的价值，即一个状态采取所有行为后的一个价值的期望值。如果能求出梯度$\nabla_{\theta}J(\theta)$，就可以用梯度上升法求解这个问题，即求解价值函数的最大值。

在这个函数中，$J(\theta)$既依赖于动作的选择有依赖于动作选择时所处状态的分布。给定一个状态，策略参数对动作选择及收益的影响可以根据参数比较直观地计算出来，但因为状态分布和环境有关，所以策略对状态分布的影响一般很难确切知道。而$J(\theta)$对模型参数的梯度却依赖于这种未知影响，那么如何估计这个梯度呢？

Sutton等人在文献中给出了这个梯度的表达式：


$$
\nabla_{\theta}J(\theta)\propto\sum_{s}\mu_{\pi_{\theta}}(s)\sum_{a}q_{\pi_{\theta}}(s,a)\nabla_{\theta}\pi_{\theta}(s,a)
$$
其中，$\mu_{\pi_{\theta}}(s)$称为策略$\pi_{\theta}$的在策略分布。在持续问题中，$\mu_{\pi_{\theta}}(s)$为算法$s_{0}$出发经过无限多步后位于状态$s$的概率。

策略梯度定理的证明：（注意：在证明过程中，为使符号简单，我们在所有公式中隐去了$\pi$对$\theta$的依赖）


$$
\scriptsize{
\begin{align}
\nabla v_{\pi}(s) &= \nabla[\sum_{a}\pi(a|s)q_{\pi}(s,a) \\
&= \sum_{a}[\nabla\pi(a|s)q_{\pi}(s,a)+\pi(a|s)\nabla q_{\pi}(s,a)] \\
&= \sum_{a}[\nabla\pi(a|s)q_{\pi}(s,a)+\pi(a|s)\nabla\sum_{s^{'},r}p(s^{'},r|s,a)(r+v_{\pi}(s^{'}))] \\
&= \sum_{a}[\nabla\pi(a|s)q_{\pi}(s,a)+\pi(a|s)\sum_{s^{'}}p(s^{'}|s,a)\nabla v_{\pi}(s^{'})] \\
&= \sum_{a}[\nabla\pi(a|s)q_{\pi}(s,a)+\pi(a|s)\sum_{s^{'}}p(s^{'}|s,a)\sum_{a^{'}}[\nabla\pi(a^{'}|s^{'})q_{\pi}(s^{'},a^{'})+\pi(a^{'}|s^{'})\sum_{s^{''}}p(s^{''}|s^{'},a^{'})\nabla v_{\pi}(s^{''})]] \\
&= \sum_{s \in S}\sum_{k=0}^{\infty}Pr(s\rightarrow x,k,\pi)\sum_{a}\nabla\pi(a|x)q_{\pi}(x,a)
\end{align}}
$$
其中，$Pr(s\rightarrow x,k,\pi)$是在策略$\pi$下，状态$s$在$k$步内转移到状态$x$的概率。所以，我们可以得到：


$$
\begin{align}
\nabla J(\theta) &= \nabla v_{\pi}(s_{0}) \\
&= \sum_{s}(\sum_{k=0}^{\infty}Pr(s_{0}\rightarrow s,k,\pi))\sum_{a}\nabla\pi(a|s)q_{\pi}(s,a) \\
&= \sum_{s}\eta(s)\sum_{a}\nabla\pi(a|s)q_{\pi}(s,a) \\
&= \sum_{s^{'}}\eta(s^{'})\sum_{s}\frac{\eta(s)}{\sum_{s^{'}}\eta(s^{'})}\sum_{a}\nabla\pi(a|s)q_{\pi}(s,a) \\
&= \sum_{s^{'}}\eta(s^{'})\sum_{s}\mu(s)\sum_{a}\nabla\pi(a|s)q_{\pi}(s,a) \\
&\propto\sum_{s}\mu(s)\sum_{a}\nabla\pi(a|s)q_{\pi}(s,a)
\end{align}
$$



# 蒙特卡洛策略梯度定理

根据策略梯度定理表达式计算策略梯度并不是一个简单的问题，其中对$\mu_{\pi_{\theta}}$和$q_{\pi_{\theta}}$的准确估计本来就是难题，更不要说进一步求解$\nabla_{\theta}J(\theta)$了。好在蒙特卡洛法能被用来估计这类问题的取值，因此首先要对策略梯度定理表达式进行如下的变形：


$$
\begin{align}
\nabla_{\theta}J(\theta) &\propto\mu_{\pi_{\theta}}(s)\sum_{a}q_{\pi_{\theta}}(s,a)\nabla_{\theta}\pi_{\theta}(s,a) \\
&= \sum_{s}\mu_{\pi_{\theta}}(s)\sum_{a}\pi_{\theta}(s,a)[q_{\pi_{\theta}}(s,a)\frac{\nabla_{\theta}\pi_{\theta}(s,a)}{\pi_{\theta}(s,a)}] \\
&= \mathbb{E}_{s, a\sim\pi}[q_{\pi_{\theta}}(s,a)\frac{\nabla_{\theta}\pi_{\theta}(s,a)}{\pi_{\theta}(s,a)}] \\
&= \mathbb{E}_{s, a\sim\pi}[q_{\pi_{\theta}}(s,a)\nabla_{\theta}\ln{\pi_{\theta}(s,a)}]
\end{align}
$$
上式为梯度策略定理的一个常见变形，但由于式中存在$q_{\pi_{\theta}}$，算法无法直接使用蒙特卡洛法来求取其中的期望。为此，进一步将该式进行变形，得到：


$$
\begin{align}
\nabla_{\theta}J(\theta) &\propto\mathbb{E}_{s, a\sim\pi}[\mathbb{E}_{T(s, a)\sim\pi}[G_{t}|s,a]\nabla_{\theta}\ln{\pi_{\theta}(s,a)}] \\
&= \mathbb{E}_{s, a\sim\pi}[G_{t}\nabla_{\theta}\ln{\pi_{\theta}(s,a)}]
\end{align}
$$
其中，$T(s,a)$表示从状态$s$开始执行动作$a$得到的一条轨迹（不包括$s$和$a$），$G_{t}$为从状态$s$开始沿着轨迹$T(s,a)$运动所得的回报。可以使用蒙特卡洛采样法来求解（即上述公式），算法只需要根据策略来采样一个状态$s$、一个动作$a$和将来的轨迹，就能构造上述公式中求取期望所对应的一个样本。

# REINFORCE 算法

REINFORCE （蒙特卡洛策略梯度） 算法是一种策略参数学习方法，其中策略参数 $\theta$ 的更新方法为梯度上升法，它的目标是为了最大化性能指标 $J(\theta)$ , 其更新公式为：


$$
\theta_{t+1} = \theta_{t} + \alpha\widehat{\nabla J(\theta_t)}
$$
根据蒙特卡洛定理中对 $\nabla_{\theta}J(\theta)$ 的计算，则有：


$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{s, a\sim\pi}[G_{t}\nabla_{\theta}\ln{\pi_{\theta}(s,a)}]
$$
根据上述梯度更新公式， 得到蒙特卡洛策略梯度更新公式：


$$
\theta = \theta + \eta\gamma^{'} G\nabla_\theta\ln\pi_\theta(s_t, a_t)
$$
其中，$\eta$ 为学习率，$\gamma^{'}$ 为衰减率，在REINFORCE算法中，暂不考虑衰减问题，设置 $\gamma^{'} = 1$ 。

**REINFORCE 算法流程：**

> 输入：马尔可夫决策过程$MDP=(S, A, P, R, \gamma)$，即状态，智能体，决策，奖励和折现系数，$\gamma = 1$，暂不讨论。
>输出：策略 $\pi(a|s, \theta)$，即在状态为s，参数为$\theta$的条件下，选择动作a的概率。
> 算法的具体流程：
> 
> 1. 随机初始化；
>  2. repeat
>    3. 根据策略$\pi_\theta$采样一个片段(episode，即智能体由初始状态不断通过动作与环境交互，直至终止状态的过程)，获得$s_0, a_0, R_1, s_1, ..., s_T-1, a_T-1, R_T$；
>       2. for $t \leftarrow 0$ to $T - 1$ do
>          1. $G \leftarrow \sum_{k=1}^{T-t} \gamma_{k-1} R_{t+k}$，G是对回报的计算，回报是奖励随时间步的积累，在本实验中，$\gamma = 1$。
>           2. $\theta = \theta + \eta\gamma^{'} G\nabla_\theta\ln\pi_\theta(s_t, a_t)$，其中$\eta$是学习率。策略梯度算法采用神经网络来拟合策略梯度函数，计算策略梯度用于优化策略网络。
> 4. 直到$\theta$收敛

