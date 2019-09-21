import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_buf import PrioritizedReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, num_states, num_actions, hidden=256):
        super(DQN, self).__init__()
        self.action_fc1 = nn.Linear(num_states, hidden)
        self.action_fc2 = nn.Linear(hidden, num_actions)
        self.state_fc1 = nn.Linear(num_states, hidden)
        self.state_fc2 = nn.Linear(hidden, 1)

    def forward(self, state):
        action_scores = self.action_fc2(F.relu(self.action_fc1(state)))
        state_scores = self.state_fc2(F.relu(self.state_fc1(state)))
        action_scores_mean = action_scores.mean(dim=1, keepdim=True)
        action_scores_centered = action_scores - action_scores_mean
        q_out = state_scores + action_scores_centered
        return q_out


def set_global_seeds(myseed):
    if myseed is not None:
        torch.manual_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)


def build_train(num_states, num_actions, lr=5e-4, gamma=1.0):
    dqn_local = DQN(num_states, num_actions).to(device)
    dqn_target = DQN(num_states, num_actions).to(device)
    optimizer = optim.Adam(dqn_local.parameters(), lr=lr)

    l1_loss = nn.L1Loss(reduction='none')
    huber_loss = nn.SmoothL1Loss(reduction='none')

    def act(state, eps):
        if random.random() <= eps:
            return random.randint(0, num_actions - 1)

        state = torch.tensor([state]).float().to(device)

        with torch.no_grad():
            action_values = dqn_local.eval()(state)
            return action_values.argmax(dim=1).item()

    def train(obs_t, act_t, rew_t, obs_tp1, dones, weights, t):
        obs_t = torch.tensor(obs_t).float().to(device)
        act_t = torch.tensor(act_t).to(device)
        rew_t = torch.tensor(rew_t).float().to(device)
        obs_tp1 = torch.tensor(obs_tp1).float().to(device)
        dones = torch.tensor(dones).float().to(device)
        weights = torch.tensor(weights).float().to(device)

        with torch.no_grad():
            best_actions = dqn_local.eval()(obs_tp1).argmax(dim=1).unsqueeze(-1)
            q_tp1 = dqn_target.eval()(obs_tp1)
            q_tp1_best = q_tp1.gather(dim=1, index=best_actions).squeeze(-1)
            q_t_selected_target = rew_t + gamma * (1.0 - dones) * q_tp1_best

        q_t = dqn_local.train()(obs_t)
        q_t_selected = q_t.gather(dim=1, index=act_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            td_error = l1_loss(q_t_selected, q_t_selected_target)
        loss = huber_loss(q_t_selected, q_t_selected_target)
        loss = (weights * loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return td_error.cpu().numpy()

    def update_target(tau=1.0):
        for target_param, local_param in zip(dqn_target.parameters(), dqn_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def load(path, train_mode=True):
        if not train_mode and os.path.exists(path):
            print("loading from {}".format(path))
            state_dict = torch.load(path, map_location=device)
            dqn_local.load_state_dict(state_dict)
        update_target()

    def save(path, train_mode=True):
        if train_mode:
            torch.save(dqn_local.state_dict(), path)

    return act, train, update_target, load, save


class LinearSchedule:
    def __init__(self, schedule_timestamps, final_p, initial_p=1.0):
        self.schedule_timestamps = schedule_timestamps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        if (self.schedule_timestamps == 0):
            return self.final_p

        fraction = min(1.0, float(t) / self.schedule_timestamps)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def learn(
    env,
    seed=None,
    lr=5e-4,
    total_timesteps=1000000,
    buffer_size=60000,
    exploration_fraction=0.05,
    exploration_final_eps=0.02,
    train_freq=1,
    batch_size=32,
    print_freq=100,
    checkpoint_freq=10000,
    checkpoint_path="checkpoints",
    learning_starts=1000,
    gamma=1.0,
    target_network_update_freq=6000,
    prioritized_replay=True,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=None,
    prioritized_replay_eps=1e-6,
    train_mode=True
):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=train_mode)[brain_name]
    num_actions = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    num_states = len(state)

    set_global_seeds(seed)

    act, train, update_target, load, save = build_train(
        num_states, num_actions,
        lr=lr, gamma=gamma
    )

    replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
    if prioritized_replay_beta_iters is None:
        prioritized_replay_beta_iters = total_timesteps
    beta_schedule = LinearSchedule(
        prioritized_replay_beta_iters,
        1.0, initial_p=prioritized_replay_beta0)

    exploration = LinearSchedule(
        int(exploration_fraction * total_timesteps),
        exploration_final_eps, initial_p=0.9)

    model_file = os.path.join(checkpoint_path, "model")

    load(model_file, train_mode=train_mode)

    episode_rewards = [0.0]
    episode_steps = 0
    saved_mean_reward = None

    for t in tqdm(range(total_timesteps)):
        eps = exploration.value(t)
        action = act(state, eps)

        env_info = env.step(action)[brain_name]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        if done:
            next_state = np.zeros(state.shape)
        else:
            next_state = env_info.vector_observations[0]

        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state
        episode_rewards[-1] += reward
        episode_steps += 1
        if done:
            env_info = env.reset(train_mode=train_mode)[brain_name]
            state = env_info.vector_observations[0]
            print("episode: {}, steps: {}, reward: {}".format(len(episode_rewards), episode_steps, episode_rewards[-1]))
            episode_rewards.append(0.0)
            episode_steps = 0

        if t > learning_starts and t % train_freq == 0:
            beta = beta_schedule.value(t)
            experience = replay_buffer.sample(batch_size, beta)
            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, t)
            new_priorities = td_errors + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > learning_starts and t % target_network_update_freq == 0:
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards) - 1
        if done and num_episodes % print_freq == 0:
            print(
                "step: {}".format(t),
                "episode: {}".format(num_episodes),
                "mean 100 episode reward: {}".format(mean_100ep_reward),
                "{}% time spent exploring".format(int(eps * 100))
            )

        if t > learning_starts and num_episodes > 100 and t % checkpoint_freq == 0:
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                print("saving model due to mean reward increase {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                saved_mean_reward = mean_100ep_reward
                save(model_file, train_mode=train_mode)

    return act
