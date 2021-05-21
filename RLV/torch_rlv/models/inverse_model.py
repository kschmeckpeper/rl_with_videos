import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class InverseModelNetwork(nn.Module):

    def __init__(self, beta, input_dims, output_dims=3, fc1_dims=64, fc2_dims=64, fc3_dims=64, name='inverse', chkpt_dir='tmp/invese'):
        super(InverseModelNetwork, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_inverse')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.q = nn.Linear(self.fc3_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.criterion = nn.MSELoss()
        self.device = 'cpu'

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


