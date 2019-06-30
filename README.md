# Project 1: Navigation

**Sergei Surovtsev**
<br/>
Udacity Deep Reinforcement Learning Nanodegree
<br/>
Class of May 2019

## Project Description

This project is introduction to Deep Reinforcement Learning. It involves implementing Deep-Q-Learning algorithm in two different flavors: with fully-observable state and from raw pixels.

We are given a simulator of 3D world where an agent need to collect bananas. Our goal is to collect yellow bananas and ignore blue bananas. 

In first flavor a state space consists of 37 dimensions and with raw pixel flavor we are given 84-by-84 px 3 channel image.

Control space consists of 4 dimensions. 

## Project Goals

* Introduction to Deep Reinforcement Learning
* Introduction to DQL-algorithm

## Technical Formulation of Problem 

* Set up environment as described in [Project Repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
* Complete Navigation.ipynb for complete state-space flavor
* [Optional] Complete Navigation_Pixels.ipynb

## Mathematical Models

In this project we are implementing Deep-Q-Learning (DQL) algorithm, that a variant of Temporal Difference (TD) learning. Key advantages of it that it is (1) Model-free (does not need model of underlying world, dynamics or rewards) and uses bootstrapping to estimate value function. 

DQN algorithm can be implemented as follows:

![segmentation-obstacles](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/dqn.png?raw=true "DQN")

In classical formulation for Q-Learning we use table (matrix) to estimate (quantized) state-action function. In DQN flavor we use neural network for it.

### NN for DQN with fully observable state

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

### NN for DQN with raw pixels

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = x.reshape(-1, 3, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
```

### Results

#### DQN from fully observable state

Converges to avg score ~ 13 around 700 episode.

```
Episode 100	Average Score: 0.53
Episode 200	Average Score: 2.08
Episode 300	Average Score: 5.25
Episode 400	Average Score: 7.69
Episode 500	Average Score: 10.35
Episode 600	Average Score: 12.78
Episode 700	Average Score: 13.20
Episode 800	Average Score: 13.09
Episode 900	Average Score: 13.04
Episode 1000	Average Score: 13.04
Episode 1100	Average Score: 12.94
Episode 1200	Average Score: 13.06
Episode 1300	Average Score: 12.93
Episode 1400	Average Score: 12.91
Episode 1500	Average Score: 13.20
Episode 1600	Average Score: 12.96
Episode 1700	Average Score: 13.12
Episode 1800	Average Score: 13.13
Episode 1900	Average Score: 12.58
Episode 2000	Average Score: 12.35
```

![DQN-1](https://github.com/cwiz/DRLND-Project-Navigation/blob/master/images/variant-1.png?raw=true "DQN")

#### DQN from raw pixels

(in progress)