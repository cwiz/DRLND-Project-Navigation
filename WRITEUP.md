# Project 1: Navigation [DRAFT]

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

### Prioritized Experience Replay [DRAFT]

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

Learning from pixels model must learn state space from raw pixels. Previous state space included velocity and acceleration, the dynamic parameters of an agent. To learn them we have to augment raw pixel state space:

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

## Hyperparameters

Current project was evaluated with following hyperparameters. Adam was used as gradient descent flavor. 

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 0.001               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```


## Results

Requirement for passing solution is getting average score over 100 episodes above 13 under 2000 episodes of training. Refer to Navigation.ipynb for details of implementation.

### Learning from ray-cast perception state-vector

![segmentation-obstacles](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/results-state-1.png?raw=true "DQN")

* [Vanilla DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-1.png)
* [Double DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-2.png)
* [Dueling DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-2.png)

### Learning from raw pixels 

Learning is substantially slower than in previous case.

## References

[1] [V Mnih et al. *Human-level control through deep reinforcement
learning*, Nature 518 529-533, 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

[2] [Hado et al. *Deep Reinforcement Learning with Double Q-learning*, Arxiv, 2015](https://arxiv.org/abs/1509.06461)

[3] [Ziyu el al. *Dueling Network Architectures for Deep Reinforcement Learning*, Arxiv, 2015](https://arxiv.org/abs/1511.06581)