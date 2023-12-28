# Brief code execution process

## Cartpole environment setup

The code consists of three classes: `Brain`, `Agent`, and `Environment`.

First, the entire cartpole environment is initialized by

```python
cp_env=Environment()
```

Once an object of this type, `cp_env`, is created, the `__init__()` function within the class will be executed.

```python
    def __init__(self):
        self.env = gym.make(ENV, render_mode="human")
        num_states = self.env.observation_space.shape[0] 
        num_actions = self.env.action_space.n 
        self.agent = Agent(num_states, num_actions) 
```

We create an env object using the `gym `library and set the rendering mode to "human" to allow real-time animation for user observation. Next, we initialize the observation space and action space. The observation space includes the position and velocity of the cart, as well as the angle and velocity of the pole. Therefore, the `observation_space` size is 4. As for the action space, since there are only two directions (left and right) in one dimension, its size is 2. We print the sizes of both to verify that the initialization is correct.

![f1](.\details_fig\f1.png)

Once the initial observation space and action space are obtained, we use these two parameters to create an `Agent` to control the cartpole. The definition of the `Agent` class is as follows:

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

At this point, only the `__init__()` function inside the class will be executed, and in actuality, it continues to create a `Brain` class.

```python
class Brain:
    
    def  __init__(self, num_states, num_actions):
        self.num_actions = num_actions #the number of CartPole actions
    
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIGITIZED**num_states, num_actions)) 
```

According to the input parameters, we generate the most important Q-matrix for Q-learning. One dimension of the Q-matrix represents actions, which are set as {left, right}, and the other dimension represents possible states. There are four parameters for the states, each of which is discretized into 6 values (`NUM_DIGITIZED = 6`), resulting in a total of `6x6x6x6 = 1296` states. Therefore, the size of the entire Q-matrix is `(1296x2)`.

At this point, we have completed most of the initialization work. Before training this Q-matrix to update the values in the table, we also need to specify some training parameters, such as:

```python
GAMMA = 0.99 #decrease rate
Learning_Rate = 0.5 #learning rate
MAX_STEPS = 200 #steps for 1 episode
NUM_EPISODES = 1000 #number of episodes
```

 `GAMMA` and `Learning_Rate` are parameters for the Bellman equation in Q-learning. These two parameters are the most important to tune during the training process.

Finally, we execute the `run()` method of `cp_env`, which will start the training.

```python
cp_env.run()
```

## Q-Learning Process

The `run()` method is defined as follows:

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

Before each training episode begins, the observation space needs to be reset (line 7). Since the time to maintain balance can theoretically be infinite, it is necessary to set a threshold to represent "reasonably balanced" during actual training. In this example, `MAX_STEPS = 200`, and when the balance can be maintained for up to the `MAX_STEPS` step, the agent is considered to have maintained balance. In each episode, the actual training part is executed from line 14 to 29. The remaining part is used for data output and visualization display purposes.

Line 14 decides the next action by actually calling `decide_action()`. This function first discretizes the various state parameters using `numpy.linspace()`, then selects an action according to the ε-greedy strategy. Here, ε (epsilon) is a small positive number, and it is used to randomly determine whether to enter the exploration state, so that the agent does not always choose the action with the highest Q value. This helps the agent balance between exploration and exploitation, ensuring that the agent does not get stuck in a local optimal solution and has the opportunity to find a better strategy.

In this example, the value of epsilon will decrease as the number of training episodes increases. This means that as the training progresses, there will be fewer occasions when randomness is needed, indicating that the agent has obtained a sufficiently stable Q-matrix by the later stages of training.

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

Once the required action is obtained from the Q-matrix, the next step is to call the gym library to execute it.

```python
observation_next, _, done, _,_ = self.env.step(action)    
```

When the value of `done` is False, it indicates that the current training episode has not ended, and we can use the Bellman equation to update the parameters of the Q-matrix.
$$
Q(s, a) = Q(s, a) + α * (reward + \gamma * max(Q(s', a'))- Q(s, a)) \tag{1}
$$

Briefly, the learning rate α and the discount factor γ jointly control the entire process. The learning rate α focuses on the overall learning speed, with a higher learning rate meaning the agent accepts new information more quickly, but it may also lead to instability and convergence issues. A lower learning rate may make the agent too conservative and difficult to adapt to changes in the environment. The discount factor γ is used to measure the importance of future rewards. A higher discount factor means the agent pays more attention to long-term rewards, while a lower discount factor prioritizes immediate rewards.

```python
    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
            Learning_Rate * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
```

In the Bellman equation, we also need to specify the `reward` for the current time step. We want the entire process to last as long as possible, so we give different reward values based on whether the Agent has completed a game and whether the game has ended.

When a game ends (`done = True`), different reward values are given based on the reason for the game ending:

- If the agent fails to keep the pole balanced (less than 195 steps), a penalty is given (`reward=-1`), indicating an unsuccessful game.
- If the game ends with the Agent successfully balancing the pole for over 195 steps, a reward is given (`reward=1`), indicating a successful game. If the game is not over, `reward=0` is given. Then, the `update_Q_table()` is called to update the value of a Q-matrix.

At the end of this time step, we update the observation space. This concludes one time step.

```python
observation = observation_next
```

The above describes the process of training the Cart-Pole using Q-learning.
