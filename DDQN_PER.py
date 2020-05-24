import numpy as np
import torch as T
from policy import PolicyNet
from prioritized_memory import Memory
from replay_memory import ReplayBuffer

class DqnPer():
    def __init__(self, lr:float, gamma:float, obs_dims,
                 num_actions:int, mem_size, mini_batchsize,
                 epsilon_dec, env_name, algo_name, replace_num=1000, 
                 epsilon=1.0, epsilon_min=0.1, checkpoint_dir='temp/dqn'):

        self.lr = lr
        self.gamma = gamma
        self.obs_dims = obs_dims
        self.num_actions = num_actions
        self.mini_batchsize = mini_batchsize
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.epsilon = epsilon
        
        self.mem_counter = 0
        self.copy_counter = 0
        self.replace_num = replace_num
        self.checkpoint_dir = checkpoint_dir
        self.mem_size = mem_size
        self.memories = Memory(mem_size)

        self.action_space = [i for i in range(self.num_actions)]

        self.learning_network = PolicyNet(lr=self.lr, 
                              num_actions=self.num_actions,
                              input_dims=self.obs_dims, 
                              name=env_name+'_'+algo_name+'_learning', 
                              checkpoint_dir=self.checkpoint_dir)

        self.target_network = PolicyNet(lr=self.lr, 
                              num_actions=self.num_actions,
                              input_dims=self.obs_dims, 
                              name=env_name+'_'+algo_name+'_target', 
                              checkpoint_dir=self.checkpoint_dir)

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min
    
    def store_memory(self, obs, action, reward, new_obs, done):
        # evaluate td error and append into memory

        q = self.learning_network(obs)[action]
        q_ = self.target_network(new_obs)
        y = reward + self.gamma * T.max(q_) * (1-int(done))
        error = abs(y - q).cpu().detach().numpy()

        # error = 0.99
        self.memories.add(error, (obs, action, reward, new_obs, done))
        self.mem_counter += 1

    def sample_memory(self):
        mini_batch, idxs, is_weights = self.memories.sample(self.mini_batchsize)
        mini_batch = np.array(mini_batch).transpose()

        states = np.stack(mini_batch[0], axis=0) #Concat list of np.arrays
        actions = mini_batch[1].astype(int)
        rewards = mini_batch[2].astype(float)
        new_states = np.stack(mini_batch[3], axis=0)
        dones = mini_batch[4].astype(bool)

        # print(f'---Actions: {type(actions[2])}---')

        states = T.tensor(states, dtype=T.float).to(self.target_network.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.target_network.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.target_network.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.target_network.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.target_network.device)

        # print(f'---States shape: {states.size()}')
        return idxs, is_weights, (states, actions, rewards, new_states, dones)

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.choice(len(self.action_space), 1)[0]
        else:
            # obs = np.array([obs])
            state = T.tensor([obs], dtype=T.float).to(self.learning_network.device)

            returns_for_actions = self.target_network.forward(state)
            action = T.argmax(returns_for_actions).cpu().detach().numpy()
        return action
    
    def learn(self):
        if self.mem_counter < self.mini_batchsize:
            return
        
        self.learning_network.optimizer.zero_grad()
        idxs, is_weights, samples = self.sample_memory()
        states, actions, rewards, new_states, dones = samples
        
        # print(f'---Actions shape: {actions.size()}')
        # print(f'---Actions: {actions}')

        indices = np.arange(self.mini_batchsize)
        q_pred = self.learning_network.forward(states)[indices, actions]
        
        q_next = self.learning_network.forward(new_states)
        actions_selected = T.argmax(q_next, dim=1) # Action selection based on online weights

        q_ = self.target_network.forward(new_states)  
        q_[dones] = 0.0   #Actions' return value are evaluated using target weights

        y = rewards + self.gamma * q_[indices, actions_selected]
        cost = self.learning_network.loss(y, q_pred)
        cost.backward()
        self.learning_network.optimizer.step()

        errors = T.abs(y - q_pred).cpu().detach().numpy()
        for i in range(self.mini_batchsize):
            idx = idxs[i]
            self.memories.update(idx, errors[i])

        self.decrement_epsilon()

        if self.copy_counter % self.replace_num == 0:
            self.copy_target_network()
        self.copy_counter += 1
        
    def copy_target_network(self):
        self.target_network.load_state_dict(self.learning_network.state_dict())

    def save_models(self):
        self.learning_network.save()
        self.target_network.save()
    
    def load_models(self):
        self.learning_network.load()
        self.target_network.load()