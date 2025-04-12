"""
input 12 node obs
output 2 node action flapping or no action
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN model for the Flappy Bird game.
    """
    def __init__(self, state_dim = 12, hidden_dim = 256, action_dim = 2) :
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x) :
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":
    # Test the DQN model
    state_dim = 12
    hidden_dim = 256
    action_dim = 2

    model = DQN(state_dim, hidden_dim, action_dim)
    print(model)

    # Create a random input tensor with the same shape as the state dimension
    x = torch.randn(10, state_dim)
    output = model(x)
    print(output)  # Output should have shape (1, action_dim)