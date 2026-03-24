import sys,gymnasium_sudoku,mlflow,random,math
import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from muzero import representation_net,dynamic_net,prediction_net,mcts_hypers,process_obs


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
        self.cat_action = lambda cell,value : torch.cat([cell,value.to(cell.device)])

    def search(self,observation,idx): # TODO fix cell value and target cell selection
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
        
        for _ in range(self.mcts_hypers.num_sim): # for n in range simulation
            path = [root]
            current_node = root
            depth = 1
            
            while current_node.is_expanded() and depth <= self.mcts_hypers.max_depth:
                action = self.ucb(current_node)
                current_node = current_node.childs[action]
                path.append(current_node)
                depth += 1
            
            # expand leaf node from parent's hidden state
            parent = path[-2]
            action = self.cat_action(target_cell,torch.tensor([action]))
            reward_n,state_n = self.dyn_net(parent.state,action.unsqueeze(0))
            target_cell,policy_n,value_n = self.pred_net(state_n) 
            
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


def test():
    rep_net = representation_net()
    dyn_net = dynamic_net()
    pred_net = prediction_net()

    chk = torch.load("./checkpoint-7500.pth",map_location="cpu")

    rep_net.load_state_dict(chk["representation_net_state"])
    dyn_net.load_state_dict(chk["dynamic_net_state"])
    pred_net.load_state_dict(chk["prediction_net_state"])

    mcts_ = mcts((rep_net,dyn_net,pred_net),mcts_hypers())

    env = gym.make("sudoku-v1",mode="easy",horizon=300,render_mode="human")
     

test()


