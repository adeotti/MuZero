import sys,gymnasium_sudoku,mlflow,random,math
import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import vmap
from torch.distributions import Categorical,Dirichlet
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass,asdict
from itertools import chain
from tqdm import tqdm


@dataclass(frozen=True)
class main_hypers:
    device: str = torch.device("cpu" if not torch.cuda.is_available() else "cuda" )
    max_steps: int = 1_000
    warmup: int = 50 #300
    env_horizon: int = 100  # 300
    batch_size: int = 10#32
    mini_batch: int = 40
    lr: int = 0.001
    k: int = 5
    l2_coeff: int = 0.1

@dataclass(frozen=True)
class mcts_hypers:
    num_sim: int = 10#100
    max_depth: int = 400
    epsilon: int = 0.25      # dirichlet
    alpha_value: int = 0.3   
    c1: int = 1.25           # ucb
    c2: int = 19652 
    gamma: int = 0.1         # backpropagation


def env(horizon=None):
    x = gym.make("sudoku-v1",mode="easy",horizon=horizon,render_mode="human")
    return x

def process_obs(x): # -> one hot encoding + mask
    x = torch.as_tensor(x).long() 
    m = (x == 0).unsqueeze(0).float()
    x = F.one_hot(x,num_classes=10).squeeze()
    x = x.permute(-1,0,1).float()
    return torch.cat([x,m.squeeze(1)],dim=0).unsqueeze(0) 


class representation_net(nn.Module): # h : state -> s^0
    def __init__(self):
        super().__init__()
        self.input_emb = nn.Parameter(torch.randn(11,64) * 0.1)
        self.conv1 = nn.LazyConv2d(32,1,1)   # 128
        self.conv2 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv3 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv4 = nn.LazyConv2d(32,3,1,1) # 256

    def forward(self,x):
        x = torch.einsum("nbrc,bo->norc",x,self.input_emb) 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # [1,32,9,9]
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
        latent_state = self.conv3(x) # [1,32,9,9]
        n = torch.cat([latent_state.flatten(1),action],dim=1)
        n = self.l1(n)
        n = self.l2(n)
        reward = self.l3(n)
        return reward,latent_state

class prediction_net(nn.Module): # f : s^k -> [p^k,v^k]
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv2 = nn.LazyConv2d(32,3,1,1) # 256
        self.conv3 = nn.LazyConv2d(32,3,1,1) # 256
        
        self.l1 = nn.LazyLinear(1024)
        self.l2 = nn.LazyLinear(9*81)

        self.pos = nn.LazyLinear(81)
        self.value = nn.LazyLinear(1)

    def forward(self,latent_state):
        B = latent_state.size(0)

        x = self.conv1(latent_state)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.l1(x.flatten(1))
        x = self.l2(x) # 729
        
        pre_pos = F.softmax(self.pos(x),-1)
        pos = Categorical(probs=pre_pos).sample()
        xpos = pos // 9
        ypos = pos % 9 
        target_cell = torch.cat([xpos,ypos])

        x_reshaped = x.reshape(B,81,9)
        policy = F.softmax(x_reshaped[torch.arange(B),pos],-1).squeeze(1)

        value = self.value(x)
        return target_cell,policy,value
    

class node:
    def __init__(self,prior):
        self.prior = prior       # P(s,a)
        self.visit_count = 0.0   # N(s,a)
        self.mean_value = 0.0    # Q(s,a)     
        self.reward = 0.0        # R(s,a)
        self.state = None        # S(s,a)
        self.childs = {}

    def is_expanded(self):
        return len(self.childs) > 0

class mcts:
    def __init__(self,networks:list,hypers):
        self.mcts_hypers = hypers
        self.rep_net,self.dyn_net,self.pred_net = networks
        self.cat_action = lambda cell,value : torch.cat([cell,value])

    def search(self,observation,idx):
        hidden_state = self.rep_net(process_obs(observation))
        target_cell,policy,value = self.pred_net(hidden_state)

        root = node(0) ; root.state = hidden_state ; depth = 0

        if not root.is_expanded(): # expand root + dirichlet noise on priors
            alpha = torch.full((9,),self.mcts_hypers.alpha_value)
            noise = Dirichlet(alpha).sample()
            for n,p in enumerate(policy.squeeze()):
                # p'(a) = (1-epsilon) * p'(a) + (epsilon * noise)
                prior = (1 - self.mcts_hypers.epsilon) * p.item() 
                prior += self.mcts_hypers.epsilon * noise[n].item() 
                root.childs[n+1] = node(round(prior,4))
            depth += 1 
        
        for _ in range(self.mcts_hypers.num_sim): # for n in range simulation
            path = [root]
            current_node = root
            
            while current_node.is_expanded():
                action = self.ucb(current_node)
                current_node = current_node.childs[action]
                path.append(current_node)
                depth+=1
            
            # expand leaf node from parent's hidden state
            parent = path[-2]
            action = self.cat_action(target_cell,torch.tensor([action]))
            reward_n,state_n = self.dyn_net(parent.state,action.unsqueeze(0))
            _,policy_n,value_n = self.pred_net(state_n) # TODO attention to the target cell 
            
            path[-1].state = state_n ; path[-1].reward = reward_n
            
            for n, p in enumerate(policy_n.squeeze()): # create childs
                path[-1].childs[n+1] = node(p.item())        
         
            for nod in reversed(path): 
                nod.visit_count += 1
                nod.mean_value += (value_n - nod.mean_value) / nod.visit_count
                value_n = nod.reward + self.mcts_hypers.gamma * value_n
             
        pi = torch.tensor([v.visit_count for v in root.childs.values()])
        pi /= pi.sum()
    
        action = torch.multinomial(pi,1).item() + 1 
        value = root.childs[action].mean_value
        return pi,action,value.squeeze(),target_cell,depth
    
    def ucb(self,parent): 
        scores = {}
        c1 = self.mcts_hypers.c1  ; c2 = self.mcts_hypers.c2 

        for action,child in parent.childs.items():
            x = child.prior
            x *= math.sqrt(parent.visit_count + 1) / (1 + child.visit_count)
            x *= c1 + math.log((parent.visit_count + c2 + 1) / c2)
            scores[action] = child.mean_value + x
        a = max(scores,key=scores.get)
        return a
        

class replay_buffer:
    def init_buffer(self):
        self.mcts_pi = torch.empty((self.hypers.max_steps,1,9),device=self.hypers.device)
        self.mcts_value = torch.empty((self.hypers.max_steps,1),device=self.hypers.device)
        self.value_target = torch.empty(*self.mcts_value.shape,device=self.hypers.device)
        self.mcts_action = torch.empty((self.hypers.max_steps,1,3),device=self.hypers.device)
        #
        self.env_reward = torch.empty((self.hypers.max_steps,1),device=self.hypers.device)
        self.env_obs = torch.empty((self.hypers.max_steps,1,9,9),device=self.hypers.device,dtype=torch.float)
        self.env_trunc = torch.empty((self.hypers.max_steps,1),device=self.hypers.device,dtype=torch.bool)

    def __init__(self,env,mcts,hypers):
        self.env = env
        self.mcts = mcts
        self.hypers = hypers
        self.obs = torch.as_tensor(self.env.reset()[0])
        self.idx = (self.obs == 0).nonzero()
        self.init_buffer()
        self.pointer = 0

    def step(self,n):
        with torch.no_grad():
            mcts_pi,mcts_action,mcts_value,target_cell,mcts_depth = self.mcts.search(self.obs,self.idx)
            action = np.append(target_cell.numpy(),mcts_action)
            state,reward,done,trunc,info = self.env.step(action)
            self.env.render()
           
            self.mcts_pi[n].copy_(mcts_pi)
            self.env_obs[n].copy_(torch.as_tensor(self.obs))
            self.mcts_value[n].copy_(mcts_value)
            self.mcts_action[n].copy_(torch.as_tensor(action))
            self.env_reward[n].copy_(reward)
            self.env_trunc[n].copy_(trunc)

            if trunc: 
                self.obs = torch.as_tensor(self.env.reset()[0])
                self.idx = (self.obs == 0).nonzero()
            else: 
                self.obs = state

            self.pointer += 1
            
            if n != 0 and n % self.hypers.batch_size == 0:
                self.compute_value_target()

        return reward,mcts_depth
    
    def compute_value_target(self): 
        with torch.no_grad():
            mcts_value = self.mcts_value[self.pointer - self.hypers.batch_size : self.pointer].squeeze()
            value_target = self.value_target[self.pointer - self.hypers.batch_size : self.pointer].squeeze()
            reward = self.env_reward[self.pointer - self.hypers.batch_size : self.pointer].squeeze()

            not_done = (1 - self.env_trunc.float()[self.pointer - self.hypers.batch_size : self.pointer])
            mask = not_done.cumprod(0)
            gamma = torch.pow(torch.full((self.hypers.batch_size,),0.2),torch.arange(self.hypers.batch_size))
            
            k_steps = 5
        
            for n in range(self.hypers.batch_size):
                k_end = min(k_steps,self.hypers.batch_size - n)
                x =  reward[n:] * gamma[n:] * mask[n:]
                x += gamma[k_end] * mcts_value[k_end] 
                value_target[n] = x.sum()

            self.value_target[self.pointer - self.hypers.batch_size : self.pointer] = value_target.unsqueeze(-1)
        
    def sample(self):
        B = self.hypers.batch_size 
        K = self.hypers.k

        start = torch.randint(0,self.hypers.env_horizon - K,(B,))  
        idx = start.unsqueeze(-1) + torch.arange(K)
    
        # sample obs and process it (apply process obs function (one hot))
        s_obs = self.env_obs[idx] # observation
        s_obs = vmap(vmap(process_obs))(s_obs)
    
        s_pi = self.mcts_pi[idx]                    # Pi
        s_action = self.mcts_action[idx]            # Action
        s_reward = self.env_reward[idx]             # Reward
        s_value = self.mcts_value[idx]              # Mcts Value
        s_value_target = self.value_target[idx]     # Value Target
        return map(torch.squeeze,(s_obs,s_pi,s_action,s_reward,s_value,s_value_target))


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
        return l2 


class main:
    def __init_nets(self):
        self.representation_net = representation_net()
        self.dynamic_net = dynamic_net()
        self.prediction_net = prediction_net()
    
        init_state = torch.empty((1,11,9,9),device=None)
        action = torch.as_tensor(self.env.action_space.sample(),device=None)
        
        hidden_state = self.representation_net(init_state)
        reward,latent_state = self.dynamic_net(hidden_state,action.unsqueeze(0))
        _,policy,value = self.prediction_net(latent_state)
        
        def init_w(layer):
            if isinstance(layer,(nn.Linear,nn.Conv2d)):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.0)

        self.representation_net.apply(init_w) 
        self.dynamic_net.apply(init_w) 
        self.prediction_net.apply(init_w)

        # self.representation_net.compile() 
        # self.dynamic_net.compile() 
        # self.prediction_net.compile()

    def __init__(self):
        self.main_hypers = main_hypers()
        self.mcts_hypers = mcts_hypers()
        self.env = env(self.main_hypers.env_horizon)

        self.__init_nets()
        self.optim = Adam(
                chain(
                    self.representation_net.parameters(),
                    self.dynamic_net.parameters(),
                    self.prediction_net.parameters()),
                lr = self.main_hypers.lr
        )
        self.mcts = mcts(
                (self.representation_net,self.dynamic_net,self.prediction_net),
                self.mcts_hypers
        ) 
        self.replay_buffer = replay_buffer(self.env,self.mcts,self.main_hypers)
        self.l2 = l2_regularization(self.representation_net,self.dynamic_net,self.prediction_net)
        
    def save(self,n):
        obj = {
            "representation_net_state":self.representation_net.state_dict(),                
            "dynamic_net_state":self.dynamic_net.state_dict(),
            "prediction_net_state":self.prediction_net.state_dict(),
            "optim_state":self.optim.state_dict()
        }
        torch.save(obj,f"./functions_states-{n}.pth")

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
            mlflow.end_run()
            mlflow.set_experiment("Muzero")
            
            with mlflow.start_run() as run:
                mlflow.log_params((asdict(self.main_hypers) | asdict(self.mcts_hypers)))

                for global_step in tqdm(range(self.main_hypers.max_steps),total=self.main_hypers.max_steps):
                    step_reward,mcts_depth = self.replay_buffer.step(global_step)
                
                    if self.replay_buffer.pointer >= self.main_hypers.warmup:
                        obs,pi,action,reward,mcts_value,value_target = self.replay_buffer.sample() 
                    
                        hidden_rep = self.representation_net(obs[:,0].float())
                        u_reward,u_value,u_policy = [],[],[]
                        for i in range(self.main_hypers.k):
                            r,s = self.dynamic_net(hidden_rep,action[:,i])
                            _,p,v = self.prediction_net(s)
                         
                            u_reward.append(r)
                            u_value.append(v)
                            u_policy.append(p)

                            hidden_rep = s # update s after each loop u_reward
                        
                        u_reward = torch.stack(u_reward).squeeze().permute(-1,0)
                        u_value = torch.stack(u_value).squeeze().permute(-1,0)
                        u_policy = torch.stack(u_policy).permute(1,0,-1)
                    
                        loss_r = F.mse_loss(u_reward,reward).mean()
                        loss_v = F.mse_loss(u_value,value_target).mean()

                        # TODO fix policy loss computation taking into consideration the fact that the policy is 
                        # constrained on another distribution
                        loss_p = -(u_policy * (pi + 1e-8).log()).sum(-1).mean()
                        total_loss = loss_r + loss_v + loss_p + (self.main_hypers.l2_coeff * self.l2())
                     
                        self.optim.zero_grad(set_to_none=True)
                        total_loss.backward()
                        self.optim.step()

                        mlflow.log_metrics(
                            {
                            "loss reward":loss_r,
                            "loss value":loss_v,
                            "loss policy":loss_p,
                            "total loss": total_loss,
                            "step reward": step_reward,
                            "MCTS depth":mcts_depth
                            },
                            step = global_step
                        )
        

if __name__ == "__main__":
    import warnings,logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """

    main().run(start=True)

