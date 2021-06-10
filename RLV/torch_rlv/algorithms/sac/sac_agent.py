import numpy as np
import torch as T
import torch.nn.functional as F
from RLV.torch_rlv.buffer.replay_buffer import ReplayBuffer
from RLV.torch_rlv.models.sac_networks import ActorNetwork, ActorNetworkDiscrete, CriticNetwork, ValueNetwork


def get_agent(env, action_space_type, experiment):
    if action_space_type == "continuous":
        return Agent(alpha=experiment.lr, beta=experiment.lr,
                     input_dims=env.observation_space.shape, env=env,
                     n_actions=env.action_space.shape[0], layers=experiment.layers)

    if action_space_type == "discrete":
        return AgentDiscrete(alpha=experiment.lr, beta=experiment.lr,
                             input_dims=env.observation_space.shape, env=env,
                             n_actions=env.action_space.n, layers=experiment.layers)


class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=None,
                 env=None, gamma=0.99, n_actions=3, max_size=1000000, tau=0.005,
                 layers=None, batch_size=256, reward_scale=2):
        if layers is None:
            layers = [256, 256]
        if input_dims is None:
            input_dims = [1]
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.memory_action_free = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, fc_dims=layers,
                                  name='actor_discrete', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, fc_dims=layers,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, fc_dims=layers,
                                      name='critic_2')
        self.value = ValueNetwork(beta, input_dims, fc_dims=layers, name='value')
        self.target_value = ValueNetwork(beta, input_dims, fc_dims=layers,
                                         name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample(state, reparameterize=False)

        return actions.gpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def replace_memory(self, new_buffer):
        self.memory = new_buffer

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self, mixed_pool=None):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        if mixed_pool is not None:
            state = np.concatenate((state, mixed_pool['state']))
            action = np.concatenate((action, mixed_pool['action']))
            reward = np.concatenate((reward, np.reshape(mixed_pool['reward'], (self.batch_size,))))
            new_state = np.concatenate((new_state, mixed_pool['next_state']))
            done = np.concatenate((done, np.reshape(mixed_pool['done_obs'], (self.batch_size,))))

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()


class AgentDiscrete:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=None,
                 env=None, gamma=0.99, n_actions=3, max_size=1000000, tau=0.005,
                 layers=None, batch_size=256, reward_scale=2):
        if layers is None:
            layers = [256, 256]
        if input_dims is None:
            input_dims = [3]
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.memory_action_free = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        if env is None:
            self.n_actions = n_actions
        else:
            self.n_actions = env.action_space.n

        self.actor = ActorNetworkDiscrete(alpha, input_dims, n_actions=n_actions, fc_dims=layers,
                                          name='actor_discrete', max_action=env.action_space.n - 1)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, fc_dims=layers,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, fc_dims=layers,
                                      name='critic_2')
        self.value = ValueNetwork(beta, input_dims, fc_dims=layers, name='value')
        self.target_value = ValueNetwork(beta, input_dims, fc_dims=layers,
                                         name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _, _ = self.actor.sample(state)
        np_action = actions.cpu().detach().numpy()
        return np.where(np_action[0] == 1)[0][0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def remember_action_free(self, state, action, reward, new_state, done):
        self.memory_action_free.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self, mixed_pool=None):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = np.reshape(reward, (self.batch_size,))
        done = np.reshape(done, (self.batch_size,))

        if mixed_pool is not None:
            state = np.concatenate((state, mixed_pool['state']))
            action = np.concatenate((action, mixed_pool['action']))
            reward = np.concatenate((reward, np.reshape(mixed_pool['reward'], (256,))))
            new_state = np.concatenate((new_state, mixed_pool['next_state']))
            done = np.concatenate((done, np.reshape(mixed_pool['done_obs'], (256,))))

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, act_probs, log_probs = self.actor.sample(state)
        act_probs = act_probs.view(-1)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = act_probs.T * (critic_value - log_probs)
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actor_loss = act_probs.T * (log_probs - critic_value)
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
