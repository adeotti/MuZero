import torch,sys,os,gymnasium_sudoku,mlflow,random,math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from torch.optim import Adam
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from itertools import chain


def env():
    x = gym.make("sudoku-v1",mode="easy",horizon=400)
    return x

def process_obs(x): # -> one hot encoding + mask
    x = torch.as_tensor(x).long() 
    m = (x == 0).unsqueeze(0).float()
    x = F.one_hot(x,num_classes=10).permute(-1,0,1).float()
    return torch.cat([x,m],dim=0).unsqueeze(0) 


def init_weights(layers):
    pass


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
        
        n = torch.cat([latent_state.flatten(1),action.unsqueeze(0)],dim=1)
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
        self.policy = nn.LazyLinear(9)
        self.value = nn.LazyLinear(1)

    def forward(self,latent_state):
        x = self.conv1(latent_state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.l1(x.flatten(1))
        x = self.l2(x)
        
        policy = self.policy(x)
        policy = F.softmax(policy,-1)

        value = self.value(x)
        return policy,value
    

class mrv: # Cell sampling with Minimum Remaining value
    def __init__(self,state):
        #self.state = torch.as_tensor(state[0]).unsqueeze(1).expand(9,9,9)
        #self.target = torch.arange(1,10).repeat(81).reshape(9,9,9)
        pass

    def get_domains(self,state):
        pass

    def sample_cell(self):
        return torch.randint(0,9,(2,))


class node:
    def __init__(self,prior):
        self.prior = prior
        self.visit_count = 0.0   # N(s,a)
        self.mean_value = 0.0    # Q(s,a)
        self.policy = 0.0        # P(s,a)
        self.reward = 0.0        # R(s,a)
        self.state = None        # S(s,a)
        self.childs = {}

    def is_expanded(self):
        return len(self.childs) > 0
    
    def expand(self,state,reward):
        self.state = state
        self.reward = reward


class mcts:
    def __init__(self,networks:list,mrv):
        self.rep_net,self.dyn_net,self.pred_net = networks
        self.mrv = mrv
        self.cat_action = lambda cell,value : torch.cat([cell,value])

    def search(self,observation,num_sim=1):
        target_cell = self.mrv.sample_cell()

        hidden_state = self.rep_net(observation)
        policy,value = self.pred_net(hidden_state)
        
        root = node(0) ; root.state = hidden_state
        depth = 0
    
        if not root.is_expanded(): # expand root + dirichlet noise on priors
            epsilon = 0.25 ; alpha = torch.full((9,),0.3)
            noise = Dirichlet(alpha).sample()
            for n,p in enumerate(policy.squeeze()):
                prior = (1 - epsilon) * p.item() + epsilon * noise[n].item()
                # p'(a) = (1-epsilon) * p'(a) + (epsilon * noise)
                root.childs[n+1] = node(round(prior,4))
            depth += 1 
        
        a = self.ucb(root)
        action = self.cat_action(target_cell,torch.tensor([a]))
        reward_n,state_n = self.dyn_net(root.state,action)
        policy_n,value_n = self.pred_net(state_n)
        n_node = node(root.childs[a].prior)
        n_node.state = state_n ; n_node.reward = reward_n ; n_node.visit_count = 1
        depht += 1
    
    def ucb(self,parent):
        scores = {}
        c1 = 0.2 ; c2 = 0.6
        for action,child in parent.childs.items():
            x = (child.mean_value + child.prior)
            x *= (math.sqrt(parent.visit_count)) / (1 + child.visit_count)
            x *= c1 + math.log((parent.visit_count + c2 + 1) / c2)  
            scores[action] = child.mean_value + x
        a = max(scores,key=scores.get)
        return a
        

class replay_buffer:
    def init_buffer(self):
        self.curr_obs = torch.empty(0,device=None)
        self.nx_obs = torch.empty(0,device=None)
        self.true_reward = torch.empty(0,device=None)
        self.pred_reward = torch.empty(0,device=None)
        self.actions = torch.empty(0,device=None)

    def __init__(self,env,mcts):
        self.mcts = mcts
        self.env = env
        self.obs = self.env.reset()[0]
        self.init_buffer()
    
    def step(self):
        with torch.no_grad():
            self.mcts.search(process_obs(self.obs))
            pass

    def sample(self):
        pass


class l2_regularization():
    def __init__(self,*networks): 
        self.weights = self.get_weights(networks)

    def __call__(self):
        return self.forward()
    
    def get_weights(self,networks):
        chnd_params = chain(*[net.parameters() for net in networks])
        weights = [params for params in chnd_params if params.ndim>1]
        return weights

    def forward(self):
        l2 = 0.0
        for n in self.weights:
            l2 += n.square().sum()
        return 0.1*l2 # TODO Update coefficient


def n_step_return(x): # value target
    pow_ = torch.arange(0,x.size(-1))
    n = torch.pow(x,pow_).sum(-1) 
    return n


class main:
    def __init_nets(self):
        self.representation_net = representation_net()
        self.dynamic_net = dynamic_net()
        self.prediction_net = prediction_net()
        """
        init_state = torch.empty((2,11,9,9),device=None)
        action = torch.as_tensor(np.stack(self.env.action_space.sample()),device=None)
        
        hidden_state = self.representation_net(init_state)
        reward,latent_state = self.dynamic_net(hidden_state,action)
        policy,value = self.prediction_net(latent_state)
        """
        # TODO : init weights and compile nets

    def __init__(self):
        self.env = env()

        self.__init_nets()
        """
        self.optim = Adam(
                chain(
                    self.representation_net.parameters(),
                    self.dynamic_net.parameters(),
                    self.prediction_net.parameters()
                ),
                lr=0.0 # TODO : update lr
        )
        """
        self.mrv = mrv(self.env.reset()[0])
        self.mcts = mcts(
                (self.representation_net,self.dynamic_net,self.prediction_net),
                self.mrv
        )
        self.replay_buffer = replay_buffer(self.env,self.mcts)
        """
        self.l2 = l2_regularization(self.representation_net,
                         self.dynamic_net,
                         self.prediction_net
        )
        """
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
    
    def run(self,start=False):
        if start:
            self.replay_buffer.step()
            #pass
        

if __name__ == "__main__":
    main().run(start=True)
    #mrv(env().reset()[0])
