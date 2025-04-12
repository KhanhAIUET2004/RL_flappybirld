import flappy_bird_gymnasium
import gymnasium
from gymnasium.wrappers import RecordVideo
from DQN import DQN
import torch
from experience_replay import ReplayBuffer
import itertools
import yaml
import random
import torch.nn as nn
import torch.optim as optim

class Agent:

    def __init__(self, hyperparameters):
        with open("hyperparamater.yaml", "r") as file:
            hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = hyperparameters_sets['FlappyBirdv0']
        self.capacity = hyperparameters['replay_buffer_size']
        self.batch_size = hyperparameters['mini_batch_size']
        self.epsilon_final = hyperparameters['epsilon_final']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.env_id = hyperparameters['env_id']
        self.gamma = hyperparameters['gamma']
        self.learning_rate = hyperparameters['learning_rate']
        self.time_to_sync = hyperparameters['time_to_sync']
        self.policy_net = DQN().to("cpu")
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.episode = hyperparameters['nums_episodes']
        self.epsilon = hyperparameters['epsilon_init']

    def run(self, render_mode):

        env = gymnasium.make(self.env_id, render_mode=render_mode, use_lidar=False)
        obs, _ = env.reset()
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        terminated = False
        while not terminated:
            # Next action:
            if random.random() < self.epsilon:
                action = env.action_space.sample()
            else:
                action = self.policy_net(obs).argmax().item()
            # Processing:
            new_obs, reward, terminated, _, info = env.step(action)
            obs  = torch.tensor(new_obs, device=self.device, dtype=torch.float32)


    def training(self, render_mode = 'rgb_array'):
        env = gymnasium.make(self.env_id, render_mode=render_mode, use_lidar=False)
        reward_per_episode = []
        epsilon_history = []
        memory = ReplayBuffer(self.capacity)
        target_net = DQN().to(self.device)
        target_net.load_state_dict(self.policy_net.state_dict())
        step_count = 0
        # Sync target network with policy network
        if step_count % self.time_to_sync == 0:
            target_net.load_state_dict(self.policy_net.state_dict())
        for episode in range(self.episode):
            print(f"Episode {episode + 1}/{self.episode}")
            obs, _ = env.reset()
            terminated = False
            episode_reward = 0
            episode_loss = 0
            while not terminated:
                # Next action:
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    tensor_obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
                    action = self.policy_net(tensor_obs).argmax().item()
                # Processing:
                new_obs, reward, terminated, _, info = env.step(action)
                episode_reward += reward          
                memory.push((obs, action, reward, new_obs, terminated))
                
                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    # optimize the model
                    episode_loss += self.optimize(target_net, self.policy_net, batch, self.optimizer)
                    step_count += 1
                    self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
                    epsilon_history.append(self.epsilon)
                    # Sync target network with policy network
                    if step_count % self.time_to_sync == 0:
                        target_net.load_state_dict(self.policy_net.state_dict())
                obs  = new_obs
            reward_per_episode.append(episode_reward)
            print(f"Episode reward: {episode_reward}")
            print(f"episode loss: {episode_loss}")
            print(f"epsilon : {self.epsilon}")

    def optimize(self, target_net, policy_net, batch, optimizer):
        # Unpack the batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

        # Convert to tensors
        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(self.device)


        current_q_values = policy_net(obs_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = target_net(next_obs_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        loss = nn.MSELoss()(current_q_values, expected_q_values)  
        # Optimize the model  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

if __name__ == "__main__":
    agent = Agent("FlappyBird-v0")
    agent.policy_net.load_state_dict(torch.load("I:/RL/flappy bird/weights.pth", map_location="cpu"))
    agent.run("human")