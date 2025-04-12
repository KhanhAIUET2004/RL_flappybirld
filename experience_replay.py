from collections import deque
import random

class ReplayBuffer:
    """
    A class to implement experience replay buffer for DQN.
    """
    def __init__(self, capacity, seed=None):
        self.buffer = deque([], maxlen=capacity)
        if seed is not None :
            random.seed(seed)

    def push(self, experience):
        """
        Add a new experience to the buffer.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)