import torch,sys,os,gymnasium_sudoku,mlflow
import torch.nn as nn
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


class representation_net(nn.Module): # h : state -> s^0
    def __init__(self):
        super().__init__()
        pass

    def forward(self,x):
        return state

class dynamic_net(nn.Module): # g : [s^k-1,a^k] -> [r^k,s^k]
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self,sk,ak):  
        return rk,sk

class prediction_net(nn.Module): # f : s^k -> [p^k,v^k]
    def __init__(self):
        super().__innit__(self)
        pass

    def forward(self,sk):
        return pk,vk


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
     
