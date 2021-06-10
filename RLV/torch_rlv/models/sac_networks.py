import os
import torch
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc_dims=None,
                 name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        if fc_dims is None:
            fc_dims = [256, 256]
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        x_dim = self.input_dims[0] + n_actions
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(x_dim, self.fc_dims[0]))
        for i in range(1, len(self.fc_dims)):
            self.net.append(nn.Linear(self.fc_dims[i-1], self.fc_dims[i]))
        self.net.append(nn.Linear(self.fc_dims[-1], 1))

        self.optimizer = optim.Adam(self.net.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q = T.cat((state, action), dim=1)
        for i in range(len(self.net)):
            q = self.net[i](q)
            if i < len(self.net)-1:
                q = F.relu(q)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc_dims=None,
                 name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        if fc_dims is None:
            fc_dims = [256, 256]
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(*self.input_dims, self.fc_dims[0]))
        for i in range(1, len(self.fc_dims)):
            self.net.append(nn.Linear(self.fc_dims[i-1], self.fc_dims[i]))
        self.net.append(nn.Linear(self.fc_dims[-1], 1))

        self.optimizer = optim.Adam(self.net.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        v = state
        for i in range(len(self.net)):
            v = self.net[i](v)
            if i < len(self.net)-1:
                v = F.relu(v)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action=1, fc_dims=None, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        if fc_dims is None:
            fc_dims = [256, 256]
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(*self.input_dims, self.fc_dims[0]))
        for i in range(1, len(self.fc_dims)):
            self.net.append(nn.Linear(self.fc_dims[i-1], self.fc_dims[i]))
        self.net.append(nn.Linear(self.fc_dims[-1], self.n_actions))
        self.net.append(nn.Linear(self.fc_dims[-1], self.n_actions))

        self.optimizer = optim.Adam(self.net.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = state
        for i in range(len(self.net)-2):
            prob = self.net[i](prob)
            prob = F.relu(prob)

        mu = self.net[-2](prob)
        sigma = self.net[-1](prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetworkDiscrete(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc_dims=None, n_actions=3,
                 name='actor_discr', chkpt_dir='tmp/sac'):
        super(ActorNetworkDiscrete, self).__init__()
        if fc_dims is None:
            fc_dims = [256, 256]
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(*self.input_dims, self.fc_dims[0]))
        for i in range(1, len(self.fc_dims)):
            self.net.append(nn.Linear(self.fc_dims[i-1], self.fc_dims[i]))
        self.net.append(nn.Linear(self.fc_dims[-1], self.n_actions))
        self.net.append(nn.Softmax(dim=1))

        self.optimizer = optim.Adam(self.net.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = state
        for i in range(len(self.net) - 1):
            prob = self.net[i](prob)
            prob = F.relu(prob)

        act_prob = T.clamp(prob, min=self.reparam_noise, max=1)
        return self.net[-1](act_prob)

    def sample(self, state):
        action_probs = self.forward(state)
        action_dist = Categorical(probs=action_probs)

        actions = action_dist.sample()
        actions = F.one_hot(actions.clone().detach(), num_classes=self.n_actions)

        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        action_probs = action_probs.sum(1, keepdim=True)
        log_action_probs = log_action_probs.sum(1, keepdim=True)

        return actions, action_probs, log_action_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
