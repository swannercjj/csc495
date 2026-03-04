
import os
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

_NO_OP = actions.FUNCTIONS.no_op.id
_PLAYER_SELF = features.PlayerRelative.SELF


class QNetwork(nn.Module):
    def __init__(self, screen_channels, screen_size, flat_size, n_actions):
        super(QNetwork, self).__init__()
        # CNN for the screen (extracts "where the units are")
        self.conv = nn.Sequential(
            nn.Conv2d(screen_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten()
        )
        
        # Calculate the size of the conv output
        # For a 64x64 input, this is usually 32 * 14 * 14
        conv_out_size = 32 * 14 * 14 
        
        # Combined Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, obs_dict):
        screen_features = self.conv(obs_dict['screen'])
        combined = torch.cat([screen_features, obs_dict['flat']], dim=1)
        return self.fc(combined)


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
                 checkpoint_interval=2,
                 checkpoint_dir=None,
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
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir

        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_state = None
        self.last_action = None

        self.episode_return = 0.0
        self.episode_step = 0
        self.episode_index = 0
        self.last_loss = None
        self.epsilon = self.eps_start

    def save_checkpoint(self, episode_index=None, checkpoint_dir=None):
        # Prefer an explicit directory passed in; fall back to the agent's own.
        base_dir = checkpoint_dir or self.checkpoint_dir
        if base_dir is None:
            base_dir = "checkpoints"
        os.makedirs(base_dir, exist_ok=True)

        if episode_index is None:
            episode_index = self.episode_index
        filename = f"dqn_ep_{episode_index}.pt"
        path = os.path.join(base_dir, filename)
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episode": episode_index,
                "steps_done": self.steps_done,
            },
            path,
        )

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
        
        # 1. Action space size
        self.n_actions = len(action_spec.functions)

        # 2. Screen specs (Spatial)
        # We chose 3 channels (Player Relative, Selected, Unit Type)
        self.screen_channels = 3 
        self.screen_size = 64 # Assuming 64x64 map
        
        # 3. Flat spec (Non-spatial)
        # The 'player' array in SC2 is always length 11
        # [player_id, minerals, vespene, food_used, food_cap, food_army, 
        #  food_workers, idle_worker_count, army_count, warp_gate_count, larva_count]
        self.flat_size = 11 

        # 4. Initialize QNetwork
        self.q_network = QNetwork(
            screen_channels=self.screen_channels,
            screen_size=self.screen_size,
            flat_size=self.flat_size,
            n_actions=self.n_actions
        ).to(self.device)

        self.target_network = QNetwork(
            self.screen_channels, self.screen_size, self.flat_size, self.n_actions
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    # def _get_obs(self, obs):
    #     """Convert PySC2 timestep observation to flat tensor for the DQN."""
    #     observation = obs.observation
    #     parts = []

    #     # Flat player features: always length n_flat so observation size is fixed
    #     n_flat = len(FLAT_FEATURES)
    #     player = np.asarray(observation.player, dtype=np.float32)
    #     flat = np.zeros(n_flat, dtype=np.float32)
    #     for f in FLAT_FEATURES:
    #         if f.index < len(player):
    #             flat[f.index] = player[f.index]
    #     parts.append(flat)

    #     # Downsampled screen (only if setup() found feature_screen in obs_spec)
    #     screen = observation['feature_screen']
    #     n_screen_flat = (self.screen_trim_h // self.screen_pool_h) * (self.screen_trim_w // self.screen_pool_w) * self.screen_shape[0]
    #     if screen is not None:
    #         screen = np.asarray(screen)
    #         h, w = screen.shape[1], screen.shape[2]
    #         pool_h, pool_w = self.screen_pool_h, self.screen_pool_w
    #         target_h = min(h, self.screen_trim_h)
    #         target_w = min(w, self.screen_trim_w)
    #         h_trim = (target_h // pool_h) * pool_h
    #         w_trim = (target_w // pool_w) * pool_w
    #         if h_trim > 0 and w_trim > 0:
    #             screen_cropped = screen[:, :h_trim, :w_trim]
    #             screen_flat = screen_cropped.reshape(screen.shape[0], h_trim // pool_h, pool_h, w_trim // pool_w, pool_w)
    #             screen_flat = screen_flat.mean(axis=(2, 4)).flatten().astype(np.float32)
    #             if screen_flat.size < n_screen_flat:
    #                 screen_flat = np.concatenate([screen_flat, np.zeros(n_screen_flat - screen_flat.size, dtype=np.float32)])
    #             elif screen_flat.size > n_screen_flat:
    #                 screen_flat = screen_flat[:n_screen_flat]
    #             parts.append(screen_flat)
    #         else:
    #             parts.append(np.zeros(n_screen_flat, dtype=np.float32))
    #     else:
    #         parts.append(np.zeros(n_screen_flat, dtype=np.float32))

    #     x = np.concatenate(parts)
    #     return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

    def _get_obs(self, obs):
        observation = obs.observation
        
        # 1. Spatial Features (The "Map")
        # Pick relevant layers: 5 is 'player_relative' (Self, Enemy, Neutral)
        # 6 is 'selected', 7 is 'unit_type'
        screen_layers = [5, 6, 7] 
        screen = np.array(observation['feature_screen'][screen_layers], dtype=np.float32)
        # Shape: (3, 64, 64)
        
        # 2. Flat Features (The "Stats")
        # Player info: [minerals, vespene, food_used, food_cap, ...]
        player_data = np.log1p(np.array(observation.player, dtype=np.float32)) # Log scale helps DQN
        
        return {
            'screen': torch.from_numpy(screen).unsqueeze(0).to(self.device),
            'flat': torch.from_numpy(player_data).unsqueeze(0).to(self.device)
        }

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
        return actions.FunctionCall(function_id, args)

    def reset(self):
        super(DQNAgent, self).reset()
        self.prev_score = np.zeros(13, dtype=np.float32)
        self.last_state = None
        self.last_action = None
        self.episode_return = 0.0
        self.episode_step = 0
        self.episode_index += 1

    def step(self, obs):
        super(DQNAgent, self).step(obs)
        step_reward = self._get_shaped_reward(obs)

        done = obs.last()

        self.episode_return += step_reward
        self.episode_step += 1

        state = self._get_obs(obs)
        available_actions = obs.observation['available_actions']

        # Store transition from previous step
        if self.last_state is not None and self.last_action is not None:
            self.memory.push(
                self.last_state, 
                self.last_action, 
                None if done else state, 
                torch.tensor([[step_reward]], device=self.device, dtype=torch.float)
            )
            self.optimize()

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0 and self.steps_done > 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Select action (epsilon-greedy with available action masking)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.epsilon = eps_threshold
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

        if done:
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(
                    {
                        "episode": self.episode_index,
                        "episode_return": self.episode_return,
                        "episode_length": self.episode_step,
                        "epsilon": self.epsilon,
                        "steps_done": self.steps_done,
                    }
                )

        return function_call

    def _get_shaped_reward(self, obs):
        current_score = np.asarray(obs.observation['score_cumulative'], dtype=np.float32)

        # Initialize tracking on the first step
        if self.episode_step == 0:
            self.prev_score = current_score
            return 0.0

        delta = current_score - self.prev_score
        self.prev_score = current_score

        # Define weights
        weights = {
            'mineral_collected': 0.005,  # Mining reward
            'vespene_collected': 0.005,
            'kill_unit': 0.1,            # Aggression reward
            'kill_building': 0.2,
            'spent_resources': 0.005,    # Reward for building/training
            'time_penalty': -0.0005,
            'win_loss': 1.0              # Game outcome
        }

        # Calculate the components
        # Index 11/12 are spent minerals/vespene
        spent_delta = delta[11] + delta[12]
        
        # Combine everything
        total_reward = (
            (delta[3] * weights['mineral_collected']) +
            (delta[4] * weights['vespene_collected']) +
            (delta[5] * weights['kill_unit']) +
            (delta[6] * weights['kill_building']) +
            (spent_delta * weights['spent_resources']) +  # Production reward
            weights['time_penalty'] +                     # Constant tick penalty
            (obs.reward * weights['win_loss'])
        )

        return float(np.clip(total_reward, -1.0, 1.0))
        
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 1. Handle "Next State" Masking (for Q-learning target)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                    device=self.device, dtype=torch.bool)
        
        # 2. Extract and stack the inputs for the current state
        # Each 's' in batch.state is now {'screen': tensor, 'flat': tensor}
        state_screens = torch.cat([s['screen'] for s in batch.state])
        state_flats = torch.cat([s['flat'] for s in batch.state])
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 3. Compute Q(s_t, a) - the model predicts Q-values for all actions
        # We pass the dictionary/tensors to model
        current_q_values = self.q_network({'screen': state_screens, 'flat': state_flats})
        state_action_values = current_q_values.gather(1, action_batch)

        # 4. Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            # Filter out the None states (terminal states)
            non_final_next_screens = torch.cat([s['screen'] for s in batch.next_state if s is not None])
            non_final_next_flats = torch.cat([s['flat'] for s in batch.next_state if s is not None])
            
            with torch.no_grad():
                # Double DQN or simple DQN: we use target_network for stability
                target_output = self.target_network({'screen': non_final_next_screens, 
                                                    'flat': non_final_next_flats})
                next_state_values[non_final_mask] = target_output.max(1)[0]

        # 5. Compute the expected Q values (Bellman Equation)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 6. Loss and Backprop
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent "exploding gradients" in deep RL
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        self.last_loss = loss.item()