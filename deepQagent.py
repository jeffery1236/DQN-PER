import numpy as np
from policy import PolicyNet
import torch as T
from replay_memory import ReplayBuffer

class Agent():
    def __init__(self, lr:float, gamma:float, state_dims,
                 num_actions:int, epsilon_min:float, epsilon_dec:float,
                 mem_size, mini_batchsize,
                 env_name, checkpoint_dir):

        self.lr = lr
        self.gamma = gamma
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.epsilon = 1.0
        self.memories = ReplayBuffer(mem_size=mem_size,
                                     state_shape=state_dims,
                                     num_actions=num_actions)
        self.mini_batchsize = mini_batchsize
        self.mem_counter = 0
        self.copy_counter = 0
        self.checkpoint_dir = checkpoint_dir

        self.learning_network = PolicyNet(lr=self.lr,
                                          num_actions=self.num_actions,
                                          input_dims=self.state_dims,
                                          name=env_name+"_dqn_learning",
                                          checkpoint_dir=self.checkpoint_dir)
        self.target_network = PolicyNet(lr=self.lr,
                                          num_actions=self.num_actions,
                                          input_dims=self.state_dims,
                                          name=env_name+"_dqn_target",
                                          checkpoint_dir=self.checkpoint_dir)

    def store_memory(self, state, action, reward, new_state, done):
        self.memories.store(state, action, reward, new_state, done)
        self.mem_counter += 1

    def sample_memory(self):
        states, actions, rewards, new_states, dones = self.memories.sample(self.mini_batchsize)

        states = T.tensor(states).to(self.target_network.device)
        actions = T.tensor(actions).to(self.target_network.device)
        rewards = T.tensor(rewards).to(self.target_network.device)
        new_states = T.tensor(new_states).to(self.target_network.device)
        dones = T.tensor(dones).to(self.target_network.device)

        # print(f'---States shape: {states.size()}')
        return states, actions, rewards, new_states, dones  

    def get_action(self, state, num_actions):
        if np.random.random() < self.epsilon:
            action = np.random.choice(num_actions, 1)[0]
        else:
            # state = np.array(state)
            # state = state.reshape(1, -1)
            state = T.tensor([state], dtype=T.float).to(self.learning_network.device)

            returns_for_actions = self.target_network.forward(state)
            action = T.argmax(returns_for_actions).detach().numpy()
        return action
    
    def learn(self):
        if self.mem_counter < self.mini_batchsize:
            return
        
        states, actions, rewards, new_states, dones = self.sample_memory()
        indices = np.arange(self.mini_batchsize)

        q_pred = self.learning_network.forward(states)[indices, actions]
        q_next = self.target_network.forward(new_states).max(dim=1)[0] 
        # dim=1 specifies take max along actions and [0] specifies taking the values instead of indices

        # print(f'---q_pred shape: {q_pred.size()}---')
        # print(f'---q_next shape: {q_next.size()}---')

        q_next[dones] = 0.0
        targets = rewards + self.gamma * q_next
        cost = self.learning_network.loss(targets, q_pred)
        cost.backward()
        self.learning_network.optimizer.step()

        self.decrement_epsilon()

        if self.copy_counter % 4 == 0:
            self.copy_target_network()
        self.copy_counter += 1
        
    def copy_target_network(self):
        self.target_network.load_state_dict(self.learning_network.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min

    def save_models(self):
        self.learning_network.save()
        self.target_network.save()

    def load_models(self):
        self.learning_network.load()
        self.target_network.load()

