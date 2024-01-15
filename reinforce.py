import argparse

# import gym
import gymnasium as gym

# from gym import wrappers
import numpy as np

# from itertools import count
# from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

gamma = 0.99
learning_rate = 1e-4
n_episodes = 1000

parser = argparse.ArgumentParser(description="PyTorch REINFORCE example")
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor (default: 0.99)",
)
parser.add_argument(
    "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
)
parser.add_argument("--render", action="store_true", help="render the environment")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="interval between training status logs (default: 10)",
)
args = parser.parse_args()


env = gym.make("CartPole-v1", render_mode="human")
# env = wrappers.Monitor(env, "tmp/cartpole", force=True)
env.reset(seed=args.seed)
torch.manual_seed(args.seed)


class Reinforce(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class Agent:
    def __init__(self):
        self.gamma = gamma
        self.policy = Reinforce()
        # self.rewards = []
        self.actions = []
        # self.saved_log_probs = []
        self.optim = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def learn(self):
        # rewards = self.rewards
        R = 0
        n = len(self.policy.rewards)
        discount_rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R 
            discount_rewards.insert(0,R)
        # for t in range(n):
        #     R = 0
        #     for r in range(t, n):
        #         R = self.policy.rewards[r] + self.gamma * R
        #     discount_rewards.append(R)
        discount_rewards = torch.tensor(discount_rewards)
        # print(discount_rewards.shape)
        mean, std = torch.mean(discount_rewards), torch.std(discount_rewards)
        discount_rewards = (discount_rewards - mean) / std
        # log_probs = torch.tensor(self.saved_log_probs)
        loss = []
        for log_prob, R in zip(self.policy.saved_log_probs, discount_rewards):
            loss.append(-log_prob * R)
        # policy_loss = -(policy.saved_log_probs * returns).sum()
        self.optim.zero_grad()
        loss = torch.cat(loss).sum()
        # loss = -(discount_rewards * log_probs).sum()

        loss.backward()
        self.optim.step()
        self.policy.rewards.clear()
        self.policy.saved_log_probs.clear()


agent = Agent()


def main():
    running_reward = 10
    for i_episode in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            # if args.render:
            env.render()
            agent.policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        agent.learn()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if i_episode % args.log_interval == 0:
            print(
                "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    i_episode, ep_reward, running_reward
                )
            )
        if running_reward > env.spec.reward_threshold:
            print(
                "Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, t)
            )
            break
    env.close()


if __name__ == "__main__":
    main()
