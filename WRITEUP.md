# Project 1: Navigation

**Sergei Surovtsev**
<br/>
Udacity Deep Reinforcement Learning Nanodegree
<br/>
Class of May 2019

## Project Description

This project is introduction to Deep Reinforcement Learning and Deep-Q-Networks (DQN) algorithm. DQN has contributed to number of breakthroughs such as superhuman performance in Atari games [1].

In this project we are using [Unity ML-Agent] Banana Collectors environment.

[![Unity ML-Agent Banana Collectors](https://img.youtube.com/vi/heVMs3t9qSk/0.jpg)](https://www.youtube.com/watch?v=heVMs3t9qSk).

**Goal**

* Collect as many yellow bananas as possible

**Observations** 

* Variant 1: Local ray-cast perception on nearby objects
* Variant 2: First-person view of agent (84x84 RGB pixels)

**Actions** 

* Move forward
* Move backward
* Rotate left
* Rotate right

**Rewards** 

* +1 on collision with yellow banana
* -1 on collision in blue banana
 
## Project Goals

* Introduction to Deep Reinforcement Learning
* Introduction to DQN Algorithm
* Testing of improved modification of original DQN Algorithm

## Technical Formulation of Problem 

* Set up environment as described in [Project Repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
* Complete Navigation.ipynb
* [Optional] Complete Navigation_Pixels.ipynb

## Mathematical Models

**Reinforcement Learning (RL)** deals with family of algorithms in which an **Agent** interacts with **Environment** and receives **Rewards** for it's **Actions**. There are two types of RL algorithms: (1) **Model-Based** where we have explicit model of environment, agent and their interactions (2) **Model-Free** in which agent has to learn these estimates of those models. Goal of agent is to maximize reward.

In this project we are dealing with **Terporal-Difference (TD)** algorithm belonging to Model-Free family of algorithms called **Deep-Q-Networks (DQN)**.

TD algorithms try to predict a metric such as **Expected Discounted Sum of Future Rewards V** that depend on future rewards that agents gets by following a **Policy P**. TD methods use bootstrapping and estimate V by sampling environment. P is a mapping from state to actions. V estimates expected sum of rewards for following P.

Deep-Q-Networks is a modification of Q-Learning algorithm which uses Neural Networks. In Q-Learning we are estimating a **Policy P** by estimating a state-action mapping Q. Classic formulation describes Q as a tabular mapping and DQN flavor uses Neural Networks.


### Vanilla DQN

**Algorithm**

<img src="https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/dqn.png?raw=true" width="450">

**Neural Network**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1              [-1, 256, 64]           2,432
            Linear-2              [-1, 256, 64]           4,160
            Linear-3               [-1, 256, 4]             260
================================================================
Total params: 6,852
Trainable params: 6,852
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 0.26
Params size (MB): 0.03
Estimated Total Size (MB): 0.32
----------------------------------------------------------------
```

### Double DQN

Improvement to DQN which samples action from target networks instead from local one by indexing target network by argmax of of local network. Intuition for this improvement is that those 2 networks have to agree on action which should improve convergence properties of the algorithm. [2]

```python
index = self.qnetwork_local.forward(next_states).detach().argmax(1)
Q_targets_next = self.qnetwork_target.forward(next_states).detach()
_a = tuple([(i, j) for i, j in enumerate(list(index))])
Q_targets_next = torch.stack([Q_targets_next[i] for i in _a])
```

### Dueling DQN

Dueling DQN changes architecture of Q-Network by splitting it's head to State value and State-Action value estimates. 

**Neural Network**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1              [-1, 256, 64]           2,432
            Linear-2              [-1, 256, 64]           4,160
            Linear-3              [-1, 256, 32]           2,080
            Linear-4               [-1, 256, 1]              33
            Linear-5              [-1, 256, 32]           2,080
            Linear-6               [-1, 256, 4]             132
================================================================
Total params: 10,917
Trainable params: 10,917
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 0.38
Params size (MB): 0.04
Estimated Total Size (MB): 0.46
----------------------------------------------------------------
```

### Prioritized Experience Replay

**TODO: Check correctness**

Prioritized Experience Replay [PER] is modification of DQN that changes the way we sample from Experience Replay buffer by assigning weights proportional to an error signal.

Here's the change to Experience buffer. We assign sampling probabilities proportional to priorities.

```python
def get_probabilities_from_priorities(self):
    priorities = np.array(
        [e.priority for e in self.memory if e is not None])
    scaled_priorities = (priorities + self.epsilon)**self.alpha
    return scaled_priorities / np.sum(scaled_priorities)

def sample(self):
    probabilities = self.get_probabilities_from_priorities()
    idxs = np.random.choice(
        np.arange(0, len(self.memory)), self.batch_size, p=probabilities)
    experiences = []
    for j, i in enumerate(idxs):
        self.memory[i].probability = probabilities[j]
        experiences.append(self.memory[i])
    ...
```

Here's how we assign priorities to experience samples in ```Agent.learn``` function.

```python
if self.priority_replay:
    td_error = (
        Q_expected - Q_targets).detach().abs().cpu().numpy().reshape(-1)
    self.memory.update_priorities(idxs, td_error)
    p = self.memory.get_probabilities_from_indices(idxs)
    p = torch.cuda.FloatTensor((1. / BATCH_SIZE) * (1. / p))
    loss = (p * loss).mean()
```

## Learning from Pixels

**State augmentation**

In this variation of the task we are only given partially-observable state space which is raw pixels from agent's view perspective. This is example of Partially observable Markov decision process in which system dynamics are determined by an MDP but agent can't observe full state space. 

In order to 'hack' this problem with DQN implementation defined for previous flavor we can augment state space to capture more latent data. By merging 3 consecutive frames and actions we can force NN to learn more data that from just single frame.

In experiment with observation space defined as a single frame average rewards per 100 episodes oscillated around. With augmentation trick I was able to get to score of 3.

**this section will be updated with recent updates to DQN for Partially observable Markov decision processes** 

```python
def augment_state(frames, actions):
    action_t_minus_1, action_t = actions[-1], actions[0]
    pix_t_minus_1, pix_t, pix_t_plus_1 = frames[0]
           
    return np.stack([
        pix_t_minus_1,    # unrolled to 3 dimensions
        action_t_minus_1, # 1 dim
        pix_t,            # unrolled to 3 dimensions
        action_t,         # 1 dim
        pix_t_plus_1,     # unrolled to 3 dimensions 
    ])
```

**Neural Network**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 40, 40]           4,416
       BatchNorm2d-2           [-1, 16, 40, 40]              32
            Conv2d-3           [-1, 32, 18, 18]          12,832
       BatchNorm2d-4           [-1, 32, 18, 18]              64
            Conv2d-5             [-1, 32, 7, 7]          25,632
       BatchNorm2d-6             [-1, 32, 7, 7]              64
            Linear-7                   [-1, 64]         100,416
            Linear-8                    [-1, 4]             260
================================================================
Total params: 143,716
Trainable params: 143,716
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 18.95
Forward/backward pass size (MB): 0.57
Params size (MB): 0.55
Estimated Total Size (MB): 20.07
----------------------------------------------------------------
```

## Results

Requirement for passing solution is getting average score over 100 episodes above 13 under 2000 episodes of training. Refer to Navigation.ipynb for details of implementation.

### Learning from ray-cast perception state-vector

#### Hyperparameters

Current project was evaluated with following hyperparameters. Adam was used as gradient descent flavor. 

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```

#### DQN Variants Comparison over 100-episode Averages

![results-summary-1](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/results-state-1.png?raw=true "DQN")

#### Rewards-per-Episode Plots

* [Vanilla DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-1.png)
* [Double DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-2.png)
* [Dueling DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-2.png)

### Learning from raw pixels 

#### Rewards-per-Episode

```
Episode 100	Average Score: 0.29
Episode 200	Average Score: 0.96
Episode 300	Average Score: 1.81
Episode 400	Average Score: 2.07
Episode 500	Average Score: 2.54
Episode 600	Average Score: 2.87
Episode 700	Average Score: 2.92
Episode 800	Average Score: 2.98
Episode 900	Average Score: 3.41
Episode 1000	Average Score: 3.74
Episode 1100	Average Score: 3.89
Episode 1200	Average Score: 3.67
Episode 1300	Average Score: 3.67
Episode 1400	Average Score: 3.57
Episode 1500	Average Score: 3.50
Episode 1600	Average Score: 4.08
Episode 1700	Average Score: 3.21
Episode 1800	Average Score: 3.38
Episode 1900	Average Score: 3.38
Episode 2000	Average Score: 3.32
Episode 2100	Average Score: 3.30
Episode 2200	Average Score: 2.66
Episode 2300	Average Score: 2.92
Episode 2400	Average Score: 3.42
Episode 2500	Average Score: 2.92
Episode 2600	Average Score: 3.21
Episode 2700	Average Score: 3.28
Episode 2800	Average Score: 3.86
Episode 2900	Average Score: 3.60
Episode 3000	Average Score: 3.50
```

* [Double DQN from pixels](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/pixels.png)


## References

[1] [V Mnih et al. *Human-level control through deep reinforcement
learning*, Nature 518 529-533, 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
<br />
[2] [Hado et al. *Deep Reinforcement Learning with Double Q-learning*, Arxiv, 2015](https://arxiv.org/abs/1509.06461)
<br />
[3] [Ziyu el al. *Dueling Network Architectures for Deep Reinforcement Learning*, Arxiv, 2015](https://arxiv.org/abs/1511.06581)