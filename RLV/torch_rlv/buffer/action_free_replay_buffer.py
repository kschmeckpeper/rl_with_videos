import numpy as np


class ActionFreeReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.target_memory = np.zeros((self.mem_size, 1))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.terminal_memory = np.zeros((self.mem_size, 1), dtype=np.bool)

    def store_transition(self, state, state_, done, target):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.target_memory[index] = target

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        target = self.target_memory[batch]
        dones = self.terminal_memory[batch]

        return states, states_, target, dones
