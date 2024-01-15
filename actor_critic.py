import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole
gamma = 0.99
learning_rate = 1e-4
n_episodes = 1000

parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
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


# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("Pong-v4", render_mode="human")
env.reset(seed=args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


def prepro(I):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    # return torch.flatten(torch.tensor(I))
    return I.astype(np.float32).ravel()


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(80 * 80, 200)

        # actor's layer
        self.action_head = nn.Linear(200, 6)

        # critic's layer
        self.value_head = nn.Linear(200, 1)

        # action buffer (log Prob)
        self.log_probs = []
        # reward buffer
        self.critic_values = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, values


class Agent:
    def __init__(self):
        self.gamma = gamma
        self.policy = ActorCritic()
        # self.rewards = []
        self.actions = []
        # self.saved_log_probs = []
        self.optim = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.log_probs.append(m.log_prob(action))
        self.policy.critic_values.append(value)
        return action.item()

    def learn(self):
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        # returns = []  # list to save the true values
        R = 0
        # self.optim.zero_grad()
        n = len(self.policy.rewards)
        discount_rewards = []
        for r in self.policy.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            discount_rewards.insert(0, R)

        discount_rewards = torch.tensor(discount_rewards)
        mean, std = torch.mean(discount_rewards), torch.std(discount_rewards)
        discount_rewards = (discount_rewards - mean) / std
        # log_probs = torch.tensor(self.saved_log_probs)
        loss = []
        for log_prob, value, R in zip(
            self.policy.log_probs, self.policy.critic_values, discount_rewards
        ):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            # value_losses.append(F.mse_loss(value, torch.tensor([R])))
            value_losses.append((torch.tensor([R]).detach() - value) ** 2 / 2)

        # policy_loss = -(policy.saved_log_probs * returns).sum()

        # loss = -(discount_rewards * log_probs).sum()
        self.optim.zero_grad()

        # sum up all the values of policy_losses and value_losses
        # loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optim.step()
        self.policy.rewards.clear()
        self.policy.log_probs.clear()
        self.policy.critic_values.clear()


def main():
    agent = Agent()
    running_reward = 10
    for i_episode in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):
            state = prepro(state)
            action = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
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
        # if running_reward > env.spec.reward_threshold:
        #     print(
        #         "Solved! Running reward is now {} and "
        #         "the last episode runs to {} time steps!".format(running_reward, t)
        #     )
        #     break
    env.close()


if __name__ == "__main__":
    main()
