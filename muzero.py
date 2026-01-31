import torch,sys,os,gymnasium_sudoku,mlflow
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


def env():
    def thunck():
        x = gym.make("sudoku-v1",mode="easy",horizon=400)
        return x
    return AsyncVectorEnv([thunck for _ in range(2)])

def process_obs(x): # -> one hot encoding + mask
    x = x.long() 
    m = (x == 0).unsqueeze(1).float()
    x = F.one_hot(x,num_classes=10).permute(0,-1,1,2).float() 
    return torch.cat([x,m],dim=1) 

@torch.no_grad()
def w_init(l):
    if isinstance(l,(nn.Conv2d,nn.Linear)):
        nn.init.orthogonal_(l.weight)
        l.bias.fill_(0.0)


class representation_net(nn.Module): # h : state -> s^0
    def __init__(self):
        super().__init__()
        # TODO : Add distributed representation layer
        self.conv1 = nn.LazyConv2d(32,1,1)   # 128
        self.conv2 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv3 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv4 = nn.LazyConv2d(32,3,1,1) # 256

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # 2.32.9.9
        return x

class dynamic_net(nn.Module): # g : [s^k-1,a^k] -> [r^k,s^k]
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv2 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv3 = nn.LazyConv2d(32,3,1,1) # 256

        self.l1 = nn.LazyLinear(1024)
        self.l2 = nn.LazyLinear(512)
        self.l3 = nn.LazyLinear(1)

    def forward(self,x,action=torch.rand(3,2)): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) # 2.32.9.9
        
        n = torch.cat([x.flatten(1),action.T],dim=1)
        n = self.l1(n)
        n = self.l2(n)
        reward = self.l3(n)
        return reward,latent_state

class prediction_net(nn.Module): # f : s^k -> [p^k,v^k]
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(32,3,1,1)
        self.conv2 = nn.LazyConv2d(32,3,1,1)
        self.conv3 = nn.LazyConv2d(32,3,1,1)
        
        self.l1 = nn.LazyLinear(1024)
        self.l2 = nn.LazyLinear(512)
        self.policy = nn.LazyLinear(10)
        self.value = nn.LazyLinear(1)

    def forward(self,latent_state):
        x = self.conv1(latent_state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.l1(x.flatten(1))
        x = self.l2(x)
        
        policy = self.policy(self.action_mask(x))
        policy = F.softmax(policy,-1)

        value = self.value(x)
        return policy,value
    
    def action_mask(self,x): # min(cell value) = 1 
        mask = torch.zeros_like(x,dtype=torch.bool)   
        mask[:,0] = True
        value = -1e8
        return torch.masked_fill(x,mask,value)


class mcts:
    def __innit__(self):
        pass

    def ucb(self):
        pass

    def selection(self):
        pass

    def expansion(self):
        pass

    def simulation(self):
        pass

    def backup(self):
        pass


class replay_buffer:
    def __init__(self):
        pass

    def store(self):
        pass

    def sample(self):
        pass


class main:
    def __innit__(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
    
    def log(self):
        pass

    def train(self):
        pass
     
