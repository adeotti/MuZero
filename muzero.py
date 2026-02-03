import torch,sys,os,gymnasium_sudoku,mlflow
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from itertools import chain


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

    def forward(self,x,action): 
        x = self.conv1(x)
        x = self.conv2(x)
        latent_state = self.conv3(x) # 2.32.9.9
        
        n = torch.cat([latent_state.flatten(1),action.T],dim=1)
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


class l2_reg(nn.Module): # L2 Regularization
    def __init__(self,n1,n2,n3):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.weights = self.get_weights(self.n1,self.n2,self.n3)
    
    def get_weights(self,n1,n2,n3):
        chnd_params = chain(n1.parameters(),n2.parameters(),n3.parameters())
        weights = [params for params in chnd_params if params.ndim>1]
        return weights

    def forward(self,x=0.0):
        for n in self.weights:
            x += n.square().sum()
        return x
    

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
    def __init_weights(self,layers):
        pass

    def __init_nets(self):
        self.representation_net = representation_net()
        self.dynamic_net = dynamic_net()
        self.prediction_net = prediction_net()

        init_state = torch.empty((2,11,9,9),device=None)
        action = torch.as_tensor(np.stack(self.env.action_space.sample()),device=None)
        
        hidden_state = self.representation_net(init_state)
        reward,latent_state = self.dynamic_net(hidden_state,action)
        policy,value = self.prediction_net(latent_state)
        
        # TODO : init weights and compile nets

    def __init__(self):
        self.env = env()
        self.__init_nets()
        self.mcts = mcts()
        self.replay_buffer = replay_buffer()

        self.optim = Adam(
                chain(
                    self.representation_net.parameters(),
                    self.dynamic_net.parameters(),
                    self.prediction_net.parameters()
                ),
                lr=0.0 # TODO : update lr
        )

        self.l2 = l2_reg(self.representation_net,
                         self.dynamic_net,
                         self.prediction_net
        )
        # TODO : compile l2
        
    def save(self):
        obj = {
            "representation_net_state":self.representation_net.state_dict(),                
            "dynamic_net_state":self.dynamic_net.state_dict(),
            "prediction_net_state":self.prediction_net.state_dict(),
            "optim_state":self.optim.state_dict()
        }
        torch.save(obj,"functions_states")

    def load(self,path):
        obj = torch.load(path)
        self.representation_net.load_state_dict(obj["representation_net_state"],strict=True)
        self.dynamic_net.load_state_dict(obj["dynamic_net_state"],strict=True)
        self.prediction_net.load_state_dict(obj["prediction_net_state"],strict=True)
        self.optim.load_state_dict(obj["optim_state"])

    def log_data(self):
        pass

    def train(self):
        pass
        
        
     
