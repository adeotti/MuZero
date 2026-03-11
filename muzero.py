import torch,sys,os,gymnasium_sudoku,mlflow,random,math
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

from torch.distributions import Categorical,Dirichlet
from torch.optim import Adam
from dataclasses import dataclass,asdict
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
from tqdm import tqdm


@dataclass(frozen=True)
class main_hypers:
    device: str = torch.device("cpu" if not torch.cuda.is_available() else "cuda" )
    max_steps: int = 1_000
    warmup: int = 400
    env_horizon: int = 400
    batch_size: int = 800
    mini_batch: int = 40
    lr: int = 0.001
    k: int = 5
    l2_coeff: int = 0.1

@dataclass(frozen=True)
class mcts_hypers:
    num_sim: int = 5
    max_depth: int = 1
    epsilon: int = 0 # dirichlet
    alpha_value: int = 0.3   
    c1: int = 1.25    # ucb
    c2: int = 19652 
    gamma: int = 0.1         # backpropagation


def env(horizon=None):
    x = gym.make("sudoku-v1",mode="easy",horizon=horizon)
    return x

def process_obs(x): # -> one hot encoding + mask
    x = torch.as_tensor(x).long() 
    m = (x == 0).unsqueeze(0).float()
    x = F.one_hot(x,num_classes=10).permute(-1,0,1).float()
    return torch.cat([x,m],dim=0).unsqueeze(0) 


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
        # Compute the minimum required value once at the start of an horizon and cache the data then just keep
        # sampling min(mrv cell) until env is done or trunc then recompute the entire mrv and cache and loop
        #self.state = torch.as_tensor(state[0]).unsqueeze(1).expand(9,9,9)
        #self.target = torch.arange(1,10).repeat(81).reshape(9,9,9)
        pass

    def get_domains(self,state):
        pass

    def sample_cell(self,env_trunc):
        if env_trunc:
            # TODO : recompute mrv entirely for the new state
            pass
        else:
            # sample min(mrv cell) from cached data
            pass
        return torch.randint(0,9,(2,))


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
    def __init__(self,networks:list,mrv,hypers):
        self.mcts_hypers = hypers
        self.rep_net,self.dyn_net,self.pred_net = networks
        self.mrv = mrv
        self.cat_action = lambda cell,value : torch.cat([cell,value])

    def search(self,observation,env_trunc):
        _mrv = self.mrv(observation)
        target_cell = _mrv.sample_cell(env_trunc)
        hidden_state = self.rep_net(process_obs(observation))
        policy,value = self.pred_net(hidden_state)

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
            policy_n,value_n = self.pred_net(state_n) 
            
            path[-1].state = state_n ; path[-1].reward = reward_n
            
            for n, p in enumerate(policy_n.squeeze()): # create childs
                path[-1].childs[n+1] = node(p.item())        
         
            for nod in reversed(path): # backpropagation
                nod.visit_count += 1
                nod.mean_value += value_n
                value_n = nod.reward + self.mcts_hypers.gamma * value_n
    
        pi = torch.rand((1,9))
        a = max(root.childs.keys(),key=lambda a: root.childs[a].visit_count)
        v = root.childs[a].mean_value
        return pi,a,v.squeeze(),target_cell,depth
    
    def ucb(self,parent):
        scores = {}
        c1 = self.mcts_hypers.c1  ; c2 = self.mcts_hypers.c2 

        for action,child in parent.childs.items():
            x = (child.mean_value + child.prior)
            x *= (math.sqrt(parent.visit_count)) / (1 + child.visit_count)
            x *= c1 + math.log((parent.visit_count + c2 + 1) / c2)  
            scores[action] = child.mean_value + x
        a = max(scores,key=scores.get)
        return a
        

class replay_buffer:
    def init_buffer(self):
        self.mcts_pi = torch.empty((self.hypers.batch_size,1,9),device=self.hypers.device)
        self.mcts_value = torch.empty((self.hypers.batch_size,1),device=self.hypers.device)
        self.mcts_action = torch.empty((self.hypers.batch_size,1,3),device=self.hypers.device)
        #
        self.env_reward = torch.empty((self.hypers.batch_size,1),device=self.hypers.device)
        self.env_obs = torch.empty((self.hypers.batch_size,1,11,9,9),device=self.hypers.device,dtype=torch.half)
        self.env_trunc = torch.empty((self.hypers.batch_size,1),device=self.hypers.device,dtype=torch.bool)

    def __init__(self,env,mcts,hypers):
        self.env = env
        self.mcts = mcts
        self.hypers = hypers
        self.obs = self.env.reset()[0]
        self.init_buffer()
        self.pointer = 0

    def step(self):
        trunc = done = False   # TODO : Fix 
        with torch.no_grad():
            for n in range(self.hypers.batch_size):
                mcts_pi,mcts_action,mcts_value,target_cell,_ = self.mcts.search(self.obs,trunc)
                action = np.append(target_cell.numpy(),mcts_action)
                state,reward,done,trunc,info = self.env.step(action)

                self.mcts_pi[n].copy_(mcts_pi)
                self.env_obs[n].copy_(process_obs(torch.as_tensor(self.obs)))
                self.mcts_value[n].copy_(mcts_value)
                self.mcts_action[n].copy_(torch.as_tensor(action))
                self.env_reward[n].copy_(reward)
                self.env_trunc[n].copy_(trunc)

                if trunc: 
                    self.obs = self.env.reset()[0]
                else: 
                    self.obs = state

                self.pointer += 1

            self.compute_value_target()
            
    def compute_value_target(self):
        self.value_target = torch.empty(*self.mcts_value.shape,device=self.hypers.device)
        gamma = torch.pow(torch.full((self.hypers.batch_size,),0.2),torch.arange(self.hypers.batch_size))
        mask = (1 - self.env_trunc.float()).cumprod(0) # Update 
        for n in range(self.hypers.batch_size):                
            self.value_target[n] = (
                (self.env_reward[n:].squeeze() * gamma[n:] * mask[n:].squeeze()) + self.mcts_value[n]
            ).sum()
        
    def sample(self):
        M = 32
        k = 5

        start = torch.randint(0,self.hypers.env_horizon - k,(M,))  
        idx = start.unsqueeze(-1) + torch.arange(k)
        
        s_obs = self.env_obs[idx].flatten(0,1) # Observation
        s_pi = self.mcts_pi[idx].flatten(0,1)                    # Pi
        s_action = self.mcts_action[idx].flatten(0,1)         # Action
        s_reward = self.env_reward[idx].flatten(0,1)             # Reward
        s_value = self.mcts_value[idx].flatten(0,1)              # Mcts Value
        s_value_target = self.value_target[idx].flatten(0,1)    # Value Target
         
        #print(s_obs.shape,s_pi.shape,s_action.shape,s_reward.shape,s_value.shape,s_value_target.shape)

        #sys.exit(s_obs[:,1].shape)
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
        policy,value = self.prediction_net(latent_state)
        
        def init_weights(layers):
            pass

        # TODO : init weights and compile nets

    def __init__(self):
        self.main_hypers = main_hypers()
        self.mcts_hypers = mcts_hypers()
        self.env = env(self.main_hypers.env_horizon)

        self.__init_nets()
        self.optim = Adam(
                chain(
                    self.representation_net.parameters(),
                    self.dynamic_net.parameters(),
                    self.prediction_net.parameters()
                ),
                lr = self.main_hypers.lr
        )
        self.mrv = mrv # Unitialized instance of the mrv class
        self.mcts = mcts(
                (self.representation_net,self.dynamic_net,self.prediction_net),
                self.mrv,
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

                for n in tqdm(range(self.main_hypers.max_steps),total=self.main_hypers.max_steps):
                    self.replay_buffer.step()

                    if self.replay_buffer.pointer >= self.main_hypers.warmup:
                        obs,pi,action,reward,mcts_value,value_target = self.replay_buffer.sample()
                    
                        hidden_rep = self.representation_net(obs.float())
                        u_reward,u_value,u_policy = [],[],[]
                        for i in range(self.main_hypers.k):
                            r,s = self.dynamic_net(hidden_rep,action[:,i].unsqueeze(-1))
                            p,v = self.prediction_net(s)

                            u_reward.append(r)
                            u_value.append(v)
                            u_policy.append(p)

                            hidden_rep = s # update s after each loop u_reward
                
                        u_reward = torch.stack(u_reward).squeeze(-1)
                        u_value = torch.stack(u_value).squeeze(-1)
                        u_policy = torch.stack(u_policy).squeeze()
                        
                        total_loss = F.mse_loss(u_reward,reward).mean()
                        total_loss += F.mse_loss(u_value,value_target).mean()
                        # total_loss += F.mse_loss(u_policy,pi.squeeze()).mean() fix
                        total_loss += self.main_hypers.l2_coeff * self.l2()
                        sys.exit()
                    
                        total_loss = torch.tensor([0.0],requires_grad=True)
                        self.optim.zero_grad(set_to_none=True)
                        total_loss.backward()
                        self.optim.step()

                        mlflow.log_metrics(
                            {
                            "_loss": 0.0,
                            "__": 0.0
                            }
                        )
                
                

            

if __name__ == "__main__":
    main().run(start=True)
    #mrv(env().reset()[0])
    #print(n_step_return(torch.tensor([2,4,2,4])))
