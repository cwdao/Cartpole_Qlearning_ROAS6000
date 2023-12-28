# 代码简要运行流程

## Cartpole 环境设置

本代码共设置三个类`Brain`,`Agent`,`Environment`。

首先，初始化整个 cartpole 环境

```python
cp_env=Environment()
```

生成这一类型的对象`cp_env`后，会执行类里的 `__init__()`函数，

```python
    def __init__(self):
        self.env = gym.make(ENV, render_mode="human")
        num_states = self.env.observation_space.shape[0] 
        num_actions = self.env.action_space.n 
        self.agent = Agent(num_states, num_actions) 
```

调用 gym 库创建一个env对象，并指定呈现模式为“human”，该模式可以实时绘制动画，方便使用者观察。接着初始化观测空间和动作空间。其中，观测空间包括cart 的位置和速度，及pole的角度和速度。因此，`observation_space`大小为4。对于动作空间，由于只有一维的左右两个方向，其大小为2。我们打印出二者的尺寸检验初始化是否正确：

![f1](C:\Users\cwdbo\OneDrive\桌面\theTEMP\学业\GKG\ROAS6000\作业\final_proj\details_fig\f1.png)

取得初始化的观测空间和动作空间后，我们用这两个参数生成`Agent` 用以控制 cartpole。`Agent` 类的定义如下

```python
class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
    
    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(
            observation, action, reward, observation_next)
 
    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action
```

此时只会执行类里的 `__init__()`函数，，实际上是继续生成一个 `Brain` 类

```python
class Brain:
    
    def  __init__(self, num_states, num_actions):
        self.num_actions = num_actions #the number of CartPole actions
    
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIGITIZED**num_states, num_actions)) 
```

根据传入的参数，我们生成Qlearning 最重要的Q矩阵，Q矩阵的一个维度是动作，设为{左，右}，另一个维度是可能的状态。对于状态，一共有四个参数，每个参数被离散化到6个值(`NUM_DIGITIZED = 6`)，因而状态一共有`6x6x6x6  = 1296` 个，整个Q矩阵的大小就是`(1296x2)`。

到此，我们已经设置了大部分初始化的工作，在训练这张Q矩阵以更新表中的各个值前，还需要指定一些训练参数，例如：

```python
GAMMA = 0.99 #decrease rate
Learning_Rate = 0.5 #learning rate
MAX_STEPS = 200 #steps for 1 episode
NUM_EPISODES = 1000 #number of episodes
```

其中， `GAMMA `和`Learning_Rate`是Q-learning的 bellman 方程的参数。在训练过程中，这两个是最主要的调试参数。

最后，我们执行`cp_env`的`run()`方法，这将启动训练。

```python
cp_env.run()
```

## Q-Learning 学习过程

`run()`方法定义如下：

```python
    def run(self):
        complete_episodes = 0 
        is_episode_final = False 
        frames = []
        
        for episode in range(NUM_EPISODES): 
            observation,_ = self.env.reset(seed=42)  
            
            for step in range(MAX_STEPS):  
                
                if is_episode_final is True:  
                    frames.append(self.env.render())
                    
                action = self.agent.get_action(observation, episode)            
                observation_next, _, done, _,_ = self.env.step(action)          
                
                if done: 
                    if step < 195:
                        reward = -1  
                        complete_episodes = 0  
                    else:   
                        reward = 1  
                        complete_episodes += 1  
                else:
                    reward = 0   
                
                self.agent.update_Q_function(observation, action, reward, observation_next)
                
                observation = observation_next
                
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break
                
            if is_episode_final is True:  
                break
                    
            if complete_episodes >= 10:
                print('succeeded for 10 times')
                is_episode_final = True
```

每次训练开始前，观测空间需要重置（line 7）。由于维持平衡的时间理论上可以无限大，实际训练时有必要设置一个阈值来代表”可以较好地平衡“。本例中`MAX_STEPS = 200`，当平衡能够持续到第`MAX_STEPS`步时，认为Agent 维持了平衡。在每个`episode`里，实际执行训练的部分在 line 14 -29 行。其余部分用于数据输出与可视化展示等用途。

line 14, 决定下一步的动作，该处实际上调用了 decide_action() , 首先将各项状态参数使用 `numpy.linespace()`离散化，接着按照ε-greedy策略选择动作。其中ε（`epsilon`）是一个小的正数，这里随机使其判断是否进入探索状态，这样就不会始终选择Q值最高的动作。这有助于Agent在探索和利用之间取得平衡，确保Agent不会陷入局部最优解，从而有机会找到更好的策略。

本例中，epsilon 的值会随着训练轮次的增加而减小，这也就意味着，越到训练后期，需要随机的场合也会越少，此时的Agent 已经获得了足够稳定的Q矩阵。

```python
    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action
    
    def decide_action(self, observation, episode):
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))
        
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
            
        return action
        
    def digitize_state(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation
        
        digitized = [
            np.digitize(cart_pos, bins = self.bins(-2.4, 2.4, NUM_DIGITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIGITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.26, 0.26, NUM_DIGITIZED)), 
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIGITIZED))
        ]
        
        return sum([x* (NUM_DIGITIZED**i) for i, x in enumerate(digitized)])
```

从Q矩阵 中获得了所需动作，下一步就是调用gym 库执行。

```python
observation_next, _, done, _,_ = self.env.step(action)    
```

当 `done` 的值输出为0时，表示本次训练尚未中止，我们可以使用 贝尔曼方程更新Q矩阵的参数
$$
Q(s, a) = Q(s, a) + α * (reward + \gamma * max(Q(s', a'))- Q(s, a)) \tag{1}
$$


简单地说，学习率 $\alpha $ 和衰减因子 $\gamma$ 共同控制着整个过程。学习率 $\alpha $ 更关注整体的学习速率，较高的学习率意味着智能体更快地接受新的信息，但也可能导致不稳定性和收敛性问题。较低的学习率则可能导致智能体过于保守，难以适应环境的变化。衰减因子 $\gamma$ 用于衡量未来奖励的重要性。较高的折扣因子意味着智能体更关注长期奖励，而较低的折扣因子更关注即时奖励。

```python
    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
            Learning_Rate * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
```

在贝尔曼方程中，我们会发现还需要指定本次时间步的 `reward`。我们希望整个过程可以坚持的越久越好，因此，根据`Agent`是否完成了一局游戏以及游戏是否结束，给予不同的reward值。

当一局游戏结束时（`done = True`），根据游戏结束的原因来给予不同的reward值。

* 若智能体未能保持杆子平衡（步数不足195步），则给予惩罚（`reward=-1`），表示游戏未能成功进行；
* 若游戏结束时，`Agent`成功保持杆子平衡超过195步，则给予奖励（`reward=1`），表示游戏成功进行。如果游戏未结束，则`reward=0`。随后，调用`update_Q_table()`以更新一个Q矩阵的值。

在这一时间步的最后，我们更新观测空间。到此，一个时间步就结束了。

```python
observation = observation_next
```

以上就是Q-learning 训练 Cart-Pole 问题的过程。
