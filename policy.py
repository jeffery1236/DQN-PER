import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import os

class PolicyNet(nn.Module):
    def __init__(self, lr, num_actions, input_dims, name, checkpoint_dir):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, num_actions)
        # This approximates the q_table values for 1 state

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'--- T.cuda.is_available(): {T.cuda.is_available()}---')

    def forward(self, input_data):
        input_data = T.tensor(input_data, dtype=T.float).to(self.device)

        layer1 = F.relu(self.fc1(input_data))
        layer2 = self.fc2(layer1)

        return layer2

    def save(self):
        print("---Saving checkpoint---")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load(self):
        print("---Loading checkpoint---")
        state_dict = T.load(self.checkpoint_file)
        self.load_state_dict(state_dict)
