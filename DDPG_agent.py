import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import random
from collections import namedtuple, deque
from model import Actor, Critic
from configparser import ConfigParser 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent_conf = ConfigParser() 
agent_conf.read('config.ini')

Buffer = int(agent_conf.get('agent','buffer'))
Batch_size = int(agent_conf.get('agent','batch_size'))
Tau = float(agent_conf.get('agent','tau'))
lr_actor =  float(agent_conf.get('agent','lr_actor'))
lr_critic =  float(agent_conf.get('agent','lr_critic'))
w_decay = float(agent_conf.get('agent','w_decay'))
Learn_every = int(agent_conf.get('agent','learn_every'))
Learn_number = int(agent_conf.get('agent','learn_number'))

def convert_to_tensor(anylist,isbool=False):
    """" convert a list of int/float to a flaot tensor """
    if isbool:
        return torch.from_numpy(np.vstack(anylist).astype(np.uint8)).float().to(device)
    else:
        return torch.from_numpy(np.vstack(anylist)).float().to(device)

class AgentDDPG():
    """Interacts with and learns from the environment"""
        
    def __init__(self,state_size,action_size,num_agents,gamma, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int) : number of agents
            gamma (float) : discount factor
            random_seed (int): random seed
        """
        self.actor_local = Actor(state_size,action_size,random_seed).to(device)
        self.actor_target = Actor(state_size,action_size,random_seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(),lr=lr_actor)
        
        self.critic_local=Critic(state_size,action_size,random_seed).to(device)
        self.critic_target=Critic(state_size,action_size,random_seed).to(device)
        self.critic_optim=optim.Adam(self.critic_local.parameters(),lr=lr_critic,weight_decay=w_decay)
        
        self.noise = OUNoise((num_agents,action_size),random_seed)
        self.replaybuff = ReplayBuffer(Buffer,action_size,Batch_size,random_seed)
        self.gamma=gamma
        np.random.seed(random_seed)
        
    def reset(self):
        self.noise.reset()
        
    def get_action(self,states):
        """Returns action that an agent can take in a given state as per current policy."""
        states  = torch.from_numpy(states).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            actions= self.actor_local(states).cpu().data.numpy()
            
        self.actor_local.train()
        actions +=self.noise.sample()
        return np.clip(actions,-1,1)
    
    def step(self,states,actions,rewards,next_states,dones,ts):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            self.replaybuff.add(state,action,reward,next_state,done)
            
            
        if (len(self.replaybuff) > Batch_size and ts % Learn_every == 0):
            for _ in range(Learn_number):
                experiences = self.replaybuff.sample()
                self.learn(experiences)      
            
    def learn(self,experiences):
        """Update policy and value network parameters using given batch of experience tuples.
    
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
         
        """
        states,actions,rewards,next_states,dones=experiences
        
        self.update_critic(states,actions, next_states,rewards,dones)                      
        self.update_actor(states)
        self.update_networks(Tau)
        
        self.noise.reset() 
     
    def update_critic(self,states, actions,next_states,rewards,dones):
        """update the critic network using the loss calculated as 
           Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        """
        
        next_actions = self.actor_target(next_states)
        pred = self.critic_target(next_states,next_actions)
        
        Q_targets= rewards + self.gamma * pred * (1 - dones)
        Q_expected = self.critic_local(states,actions)
        
        #critic loss
        
        critic_loss = F.mse_loss(Q_expected,Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()
    
    def update_actor(self,states):
        """update the actor network using the critics Qvalue for (state,action) pair """
        
        actions_actor = self.actor_local(states)
        loss = -self.critic_local(states,actions_actor).mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        
    def update_networks(self,Tau):
        """update weights of actor critic local network
        
        """        
        self.soft_update(self.actor_local,self.actor_target,Tau)
        self.soft_update(self.critic_local,self.critic_target,Tau)
   
        
    def soft_update(self,local_model,target_model,Tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for local_params, target_params in zip(local_model.parameters(),target_model.parameters()):
            target_params.data.copy_(local_params*Tau + (1-Tau)*target_params)
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self,size,sA,batch,seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.dq=deque(maxlen=size)
        self.batch=batch
        self.action_size=sA
        self.experience=namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        np.random.seed(seed)
        
    def add(self,state,action,reward,next_state,done):
        """Add a new experience to memory."""
        experience=self.experience(state,action,reward,next_state,done)
        self.dq.append(experience)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experience=random.sample(self.dq,k=self.batch)
        
        states = convert_to_tensor([e.state for e in experience if e is not None])
        actions = convert_to_tensor([e.action for e in experience if e is not None])
        rewards = convert_to_tensor([e.reward for e in experience if e is not None])
        next_states = convert_to_tensor([e.next_state for e in experience if e is not None])
        dones = convert_to_tensor([e.done for e in experience if e is not None],isbool=True)
        
        return states,actions,rewards,next_states,dones    
    
    def __len__(self):
        """returns the current length of the replay buffer"""
        return len(self.dq)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size,seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        
        Params:
        =======
             size (int or tuple): sample space 
             mu (float): mean
             theta (float):optimal parameter
             sigma (float) :variance
        """       
        
        self.size=size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        np.random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        
        self.state = x + dx
        return self.state
    
    
    