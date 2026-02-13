
import random
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents.base_agent import BaseAgent

from agent.replay import ReplayMemory, Transition
from agent.utils import FLAT_FEATURES
import numpy as np

_NO_OP = actions.FUNCTIONS.no_op.id
_PLAYER_SELF = features.PlayerRelative.SELF


class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BaseAgent):
    def __init__(self,
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 memory_size=10000,
                 eps_start=1.0,
                 eps_end=0.05,
                 eps_decay=10000,
                 target_update_freq=1000,
                 ):
        super(DQNAgent, self).__init__()

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(memory_size)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq

        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_state = None
        self.last_action = None

    def _get_from_spec(self, spec, key):
        """Get value from obs_spec whether it's dict-like or has the key as attr."""
        if spec is None:
            return None
        try:
            return spec[key]
        except (KeyError, TypeError):
            pass
        return getattr(spec, key, None)

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self.n_actions = len(action_spec.functions)

        # Compute observation size from PySC2 obs_spec: flat player features + optional flattened screen
        n_flat = len(FLAT_FEATURES)
        feat_screen = self._get_from_spec(obs_spec, 'feature_screen')
        if feat_screen is not None:
            if isinstance(feat_screen, (tuple, list)):
                n_screen, h, w = feat_screen[0], feat_screen[1], feat_screen[2]
            else:
                n_screen, h, w = feat_screen.shape[0], feat_screen.shape[1], feat_screen.shape[2]
            self.screen_pool_h, self.screen_pool_w = max(1, h // 8), max(1, w // 8)
            self.screen_trim_h = (h // self.screen_pool_h) * self.screen_pool_h
            self.screen_trim_w = (w // self.screen_pool_w) * self.screen_pool_w
            n_screen_flat = n_screen * (self.screen_trim_h // self.screen_pool_h) * (self.screen_trim_w // self.screen_pool_w)
            self.screen_shape = (n_screen, h, w)
        else:
            self.screen_pool_h = self.screen_pool_w = 0
            n_screen_flat = 0
            self.screen_shape = None
        self.n_observations = n_flat + n_screen_flat

        self.q_network = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_network = QNetwork(self.n_observations, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def _get_obs(self, obs):
        """Convert PySC2 timestep observation to flat tensor for the DQN."""
        observation = obs.observation
        parts = []

        # Flat player features: always length n_flat so observation size is fixed
        n_flat = len(FLAT_FEATURES)
        player = np.asarray(observation.player, dtype=np.float32)
        flat = np.zeros(n_flat, dtype=np.float32)
        for f in FLAT_FEATURES:
            if f.index < len(player):
                flat[f.index] = player[f.index]
        parts.append(flat)

        # Downsampled screen (only if setup() found feature_screen in obs_spec)
        screen = observation['feature_screen']
        n_screen_flat = (self.screen_trim_h // self.screen_pool_h) * (self.screen_trim_w // self.screen_pool_w) * self.screen_shape[0]
        if screen is not None:
            screen = np.asarray(screen)
            h, w = screen.shape[1], screen.shape[2]
            pool_h, pool_w = self.screen_pool_h, self.screen_pool_w
            target_h = min(h, self.screen_trim_h)
            target_w = min(w, self.screen_trim_w)
            h_trim = (target_h // pool_h) * pool_h
            w_trim = (target_w // pool_w) * pool_w
            if h_trim > 0 and w_trim > 0:
                screen_cropped = screen[:, :h_trim, :w_trim]
                screen_flat = screen_cropped.reshape(screen.shape[0], h_trim // pool_h, pool_h, w_trim // pool_w, pool_w)
                screen_flat = screen_flat.mean(axis=(2, 4)).flatten().astype(np.float32)
                if screen_flat.size < n_screen_flat:
                    screen_flat = np.concatenate([screen_flat, np.zeros(n_screen_flat - screen_flat.size, dtype=np.float32)])
                elif screen_flat.size > n_screen_flat:
                    screen_flat = screen_flat[:n_screen_flat]
                parts.append(screen_flat)
            else:
                parts.append(np.zeros(n_screen_flat, dtype=np.float32))
        else:
            parts.append(np.zeros(n_screen_flat, dtype=np.float32))

        x = np.concatenate(parts)
        return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

    def _action_to_function_call(self, function_id, obs):
        """Convert function_id to PySC2 FunctionCall with valid args."""
        func = self.action_spec.functions[function_id]
        args = []
        for arg in func.args:
            sizes = arg.sizes
            if len(sizes) == 2:
                if obs.observation['feature_screen'] is not None:
                    h, w = obs.observation['feature_screen'].shape[1], obs.observation['feature_screen'].shape[2]
                else:
                    h, w = 64, 64
                args.append([w // 2, h // 2])
            else:
                args.append([np.random.randint(0, s) for s in sizes])
        print(function_id)
        return actions.FunctionCall(function_id, args)

    def reset(self):
        super(DQNAgent, self).reset()
        self.last_state = None
        self.last_action = None

    def step(self, obs):
        super(DQNAgent, self).step(obs)
        reward = obs.reward
        if reward > 0:
            print(reward)
        done = obs.last()

        state = self._get_obs(obs)
        available_actions = obs.observation['available_actions']

        # Store transition from previous step
        if self.last_state is not None and self.last_action is not None:
            next_state = None if done else state
            self.memory.push(self.last_state, self.last_action, next_state, torch.tensor([[reward]], device=self.device, dtype=torch.float))
            self.optimize()

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0 and self.steps_done > 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Select action (epsilon-greedy with available action masking)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values = self.q_network(state)
                # Mask invalid actions
                mask = torch.full((1, self.n_actions), -1e9, device=self.device)
                for a in available_actions:
                    mask[0, a] = 0
                action_idx = (q_values + mask).argmax(1).item()
                function_id = action_idx
        else:
            function_id = np.random.choice(available_actions)

        function_call = self._action_to_function_call(function_id, obs)

        # Store for next transition
        self.last_state = state
        self.last_action = torch.tensor([[function_id]], device=self.device, dtype=torch.long)

        return function_call

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.q_network(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
