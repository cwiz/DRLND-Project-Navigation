# Project 1: Navigation [DRAFT]

**Sergei Surovtsev**
<br/>
Udacity Deep Reinforcement Learning Nanodegree
<br/>
Class of May 2019

## Project Description

This project is introduction to Deep Reinforcement Learning and Deep-Q-Networks (DQN) algorithm. DQN has contributed to number of breakthroughs such as superhuman performance in Atari games [1] and AlphaGo.

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

Deep-Q-Networks is a modification of Q-Learning algorithm which uses Neural Networks. In Q-Learning we are estimating a **Policy P** by estimating a state-action mapping Q. Classic formulation describes Q as a tabular mapping and DQN flavor used Neural Network to learn this mapping. DQN Algorithm:


### Vanilla DQN

**Algorithm**

![segmentation-obstacles](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/dqn.png?raw=true )

**Neural Network**

```python
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
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

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc_a_units=32, fc_v_units=32):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.fc_h_a = nn.Linear(fc2_units, fc_a_units)
        self.fc_z_a = nn.Linear(fc_a_units, action_size)

        self.fc_h_v = nn.Linear(fc2_units, fc_v_units)
        self.fc_z_v = nn.Linear(fc_v_units, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        v = F.relu(self.fc_h_v(x))
        v = self.fc_z_v(v)

        a = F.relu(self.fc_h_a(x))
        a = self.fc_z_a(a)

        q = v + a - a.mean()
        return q
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

## Hyperparameters

Current project was evaluated with following hyperparameters. Adam was used as gradient descent flavor. 

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```


## Results

Requirement for passing solution is getting average score over 100 episodes above 13 under 2000 episodes of training. Refer to Navigation.ipynb for details of implementation.

### Learning from ray-cast perception state-vector

![segmentation-obstacles](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/results-state-1.png?raw=true "DQN")

* [Vanilla DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-1.png)
* [Double DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-2.png)
* [Dueling DQN Rewards-Per-Episode](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-2.png)

### Learning from raw pixels [DRAFT]

**TODO: Implement**

## References

[1] [V Mnih et al. *Human-level control through deep reinforcement
learning*, Nature 518 529-533, 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

[2] [Hado et al. *Deep Reinforcement Learning with Double Q-learning*, Arxiv, 2015](https://arxiv.org/abs/1509.06461)

[3] [Ziyu el al. *Dueling Network Architectures for Deep Reinforcement Learning*, Arxiv, 2015](https://arxiv.org/abs/1511.06581)