
import os
import random
from torch import nn
import torch.optim as optim
import torch

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents.base_agent import BaseAgent

from agent.replay import ReplayMemory, Transition
from agent.artifact_store import load_torch_state, save_torch_state
from agent.alphastar.encoders import SpatialEncoder, ScalarEncoder
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

_NO_OP = actions.FUNCTIONS.no_op.id
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = getattr(features.PlayerRelative, "NEUTRAL", 3)

SCALAR_HIDDEN_DIM = 128  # Output dim of ScalarEncoder
SPATIAL_HIDDEN_DIM = 256 # Output dim of SpatialEncoder projection

class QNetwork(nn.Module):
    """
    Q-network backed by AlphaStar encoders.

    Observation streams
    -------------------
    obs_dict['spatial'] : (B, 8, H, W)  – 8 screen feature planes
    obs_dict['scalar']  : (B, 11)       – player stats vector

    Encoding
    --------
    SpatialEncoder  →  (B, SPATIAL_HIDDEN_DIM=256)  ResBlock CNN
    ScalarEncoder   →  (B, SCALAR_HIDDEN_DIM=128)   log1p + LayerNorm MLP

    Head
    ----
    cat([spatial_enc, scalar_enc]) → (B, 384) → Linear(384, 256) → Linear(256, n_actions)
    """

    def __init__(self, n_actions: int):
        super(QNetwork, self).__init__()
        self.spatial_encoder = SpatialEncoder()   # 8 ch → (B, 256)
        self.scalar_encoder  = ScalarEncoder()    # 11-dim → (B, 128)

        combined_dim = SPATIAL_HIDDEN_DIM + SCALAR_HIDDEN_DIM  # 384
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs_dict: dict) -> torch.Tensor:
        _, spatial_enc = self.spatial_encoder(obs_dict['spatial'])  # (B, 256)
        scalar_enc     = self.scalar_encoder(obs_dict['scalar'])    # (B, 128)
        combined = torch.cat([spatial_enc, scalar_enc], dim=1)      # (B, 384)
        return self.fc(combined)                                    # (B, n_actions)


class DQNObservationPreprocessor:
    """Build model inputs from raw PySC2 observations for DQN."""

    def __init__(self, device: torch.device):
        self.device = device

    def preprocess(self, timestep):
        observation = timestep.observation

        screen = observation['feature_screen']
        planes = np.stack([
            screen.player_relative,
            screen.unit_type,
            screen.selected,
            screen.unit_hit_points_ratio,
            screen.unit_shields_ratio,
            screen.unit_density,
            screen.creep,
            screen.height_map,
        ], axis=0).astype(np.float32)

        player_data = np.asarray(observation.player, dtype=np.float32)
        player_data = np.nan_to_num(player_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'spatial': torch.from_numpy(planes).unsqueeze(0).to(self.device),
            'scalar': torch.from_numpy(player_data).unsqueeze(0).to(self.device),
        }


def _is_spatial_arg(arg_spec) -> bool:
    """Return True when an action argument represents a screen/minimap point."""
    sizes = getattr(arg_spec, "sizes", ())
    return len(sizes) == 2 and all(size > 1 for size in sizes)


def _select_point_from_layers(layers):
    """Choose a point from feature layers using a simple neutral-object heuristic."""
    player_relative = np.asarray(layers.player_relative)
    height, width = player_relative.shape

    candidate_mask = player_relative == _PLAYER_NEUTRAL
    if not candidate_mask.any():
        candidate_mask = player_relative != _PLAYER_SELF

    coords = np.argwhere(candidate_mask)
    if coords.size == 0:
        return width // 2, height // 2

    y, x = coords.mean(axis=0)
    x = int(np.clip(round(x), 0, width - 1))
    y = int(np.clip(round(y), 0, height - 1))
    return x, y


class DQNAgent(BaseAgent):
    def __init__(self,
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 memory_size=10000,
                 eps_start=1.0,
                 eps_end=0.05,
                 eps_decay=10000,
                 target_update_freq=250,
                 checkpoint_interval=100,
                 checkpoint_dir=None,
                 resume_checkpoint=None,
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
        self.resume_checkpoint = resume_checkpoint

        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_preprocessor = DQNObservationPreprocessor(device=self.device)

        self.last_state = None
        self.last_action = None
        self.last_available_actions = None

        self.episode_return = 0.0
        self.raw_episode_return = 0.0
        self.episode_score = 0.0
        self.episode_step = 0
        self.episode_index = 0
        self.last_loss = None
        self.epsilon = self.eps_start
        self.prev_self_health = 0.0

    def save_checkpoint(self, episode_index=None, checkpoint_dir=None):
        # Prefer an explicit directory passed in; fall back to the agent's own.
        base_dir = checkpoint_dir or self.checkpoint_dir
        if base_dir is None:
            base_dir = "checkpoints"

        if episode_index is None:
            episode_index = self.episode_index
        filename = f"dqn_ep_{episode_index}.pt"
        state = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode_index,
            "steps_done": self.steps_done,
        }
        save_torch_state(state, base_dir, filename)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = load_torch_state(checkpoint_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_index = checkpoint.get("episode", 0)
        self.steps_done = checkpoint.get("steps_done", 0)

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

        # QNetwork uses AlphaStar encoders internally;
        # no need to pass explicit screen_channels / flat_size.
        self.q_network = QNetwork(n_actions=self.n_actions).to(self.device)
        self.target_network = QNetwork(n_actions=self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        if self.resume_checkpoint:
            self.load_checkpoint(self.resume_checkpoint)

    def _get_obs(self, obs):
        return self.obs_preprocessor.preprocess(obs)

    def _select_spatial_point(self, obs, arg_spec):
        """Pick a plausible target point for screen/minimap click arguments.

        For mini-games, a useful default is the centroid of neutral map objects.
        If no neutral targets are visible, fall back to the center of the screen.
        """
        if getattr(arg_spec, "name", "") == "minimap":
            layers = obs.observation.get('feature_minimap')
        else:
            layers = obs.observation.get('feature_screen')

        if layers is None:
            return [0, 0]

        x, y = _select_point_from_layers(layers)

        if getattr(arg_spec, "name", "") == "screen2":
            height, width = np.asarray(layers.player_relative).shape
            x = min(width - 1, x + 1)
            y = min(height - 1, y + 1)

        return [x, y]

    def _action_to_function_call(self, function_id, obs):
        """Convert a high-level function choice into a full PySC2 action.

        The DQN selects the function id. Spatial arguments are filled by a
        separate low-level selector so the action becomes hierarchical instead
        of a single flat discrete choice.
        """
        func = self.action_spec.functions[function_id]
        args = []
        for arg in func.args:
            if _is_spatial_arg(arg):
                args.append(self._select_spatial_point(obs, arg))
            else:
                args.append([np.random.randint(0, size) for size in arg.sizes])
        return actions.FunctionCall(function_id, args)

    def reset(self):
        super(DQNAgent, self).reset()
        self.prev_score = np.zeros(13, dtype=np.float32)
        self.prev_self_health = 0.0
        self.last_state = None
        self.last_action = None
        self.last_available_actions = None
        self.episode_return = 0.0
        self.raw_episode_return = 0.0
        self.episode_score = 0.0
        self.episode_step = 0
        self.episode_index += 1

    def step(self, obs):
        super(DQNAgent, self).step(obs)
        step_reward = self._get_shaped_reward(obs)
        raw_step_reward = float(obs.reward)

        done = obs.last()

        self.episode_return += step_reward
        self.raw_episode_return += raw_step_reward
        self.episode_step += 1

        state = self._get_obs(obs)
        available_actions = obs.observation['available_actions']

        # Store transition from previous step
        if self.last_state is not None and self.last_action is not None:
            self.memory.push(
                self.last_state, 
                self.last_action, 
                None if done else state, 
                torch.tensor([[step_reward]], device=self.device, dtype=torch.float),
                self.last_available_actions,
                None if done else available_actions,
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
        self.last_available_actions = available_actions

        if done:
            current_score = np.asarray(obs.observation['score_cumulative'], dtype=np.float32)
            if current_score.size > 0:
                self.episode_score = float(current_score[0])
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(
                    {
                        "episode_return": float(self.episode_return),
                        "raw_episode_return": float(self.raw_episode_return),
                        "episode_score": float(self.episode_score),
                        "episode_length": int(self.episode_step),
                        "steps_done": int(self.steps_done),
                        "last_loss": float(self.last_loss) if self.last_loss is not None else 0.0,
                        "outcome": raw_step_reward
                    }
                )

        return function_call

    def _get_shaped_reward(self, obs):
        current_score = np.asarray(obs.observation['score_cumulative'], dtype=np.float32)
        current_self_health = self._get_total_self_health(obs)

        # Initialize tracking on the first step
        if self.episode_step == 0:
            self.prev_score = current_score
            self.prev_self_health = current_self_health
            return 0.0

        delta = current_score - self.prev_score
        self.prev_score = current_score

        health_delta = current_self_health - self.prev_self_health
        self.prev_self_health = current_self_health

        # Define weights
        weights = {
            'mineral_collected': 0.005,  # Mining reward
            'vespene_collected': 0.005,
            'kill_unit': 0.1,            # Aggression reward
            'kill_building': 0.2,
            'spent_resources': 0.005,    # Reward for building/training
            'health_loss': 0.02,         # Penalty when own unit health/shields decrease
            'time_penalty': -0.0005,
            'win_loss': 1.0              # Game outcome
        }

        # Calculate the components
        # Index 11/12 are spent minerals/vespene
        spent_delta = delta[11] + delta[12]
        health_loss_penalty = min(0.0, health_delta) * weights['health_loss']
        
        # Combine everything
        total_reward = (
            (delta[7] * weights['mineral_collected']) +
            (delta[8] * weights['vespene_collected']) +
            (delta[5] * weights['kill_unit']) +
            (delta[6] * weights['kill_building']) +
            (spent_delta * weights['spent_resources']) +  # Production reward
            health_loss_penalty +                         # Damage taken penalty
            weights['time_penalty'] +                     # Constant tick penalty
            (obs.reward * weights['win_loss'])
        )

        return float(np.clip(total_reward, -1.0, 1.0))

    def _get_total_self_health(self, obs):
        """Estimate total allied health from feature screen planes."""
        screen = obs.observation['feature_screen']
        if screen is None:
            return 0.0

        player_relative = np.asarray(screen.player_relative)
        hp_ratio = np.asarray(screen.unit_hit_points_ratio, dtype=np.float32)
        shield_ratio = np.asarray(screen.unit_shields_ratio, dtype=np.float32)

        # PySC2 ratio planes are commonly [0, 255]; normalise if needed.
        if hp_ratio.max(initial=0.0) > 1.0:
            hp_ratio = hp_ratio / 255.0
        if shield_ratio.max(initial=0.0) > 1.0:
            shield_ratio = shield_ratio / 255.0

        self_mask = (player_relative == _PLAYER_SELF)
        total_health = (hp_ratio + shield_ratio)[self_mask].sum()
        return float(total_health)
        
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 1. Handle "Next State" Masking (for Q-learning target)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                    device=self.device, dtype=torch.bool)
        
        # 2. Extract and stack the inputs for the current state
        # Each 's' in batch.state is now {'spatial': tensor, 'scalar': tensor}
        state_spatials = torch.cat([s['spatial'] for s in batch.state])
        state_scalars  = torch.cat([s['scalar']  for s in batch.state])
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).view(-1)

        # 3. Compute Q(s_t, a)
        current_q_values = self.q_network({'spatial': state_spatials, 'scalar': state_scalars})
        state_action_values = current_q_values.gather(1, action_batch)

        # 4. Compute V(s_{t+1}) for all non-terminal next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            non_final_next_spatials = torch.cat([s['spatial'] for s in batch.next_state if s is not None])
            non_final_next_scalars  = torch.cat([s['scalar']  for s in batch.next_state if s is not None])
            non_final_next_available_actions = [
                actions_for_state
                for actions_for_state in batch.next_available_actions
                if actions_for_state is not None
            ]
            
            with torch.no_grad():
                target_output = self.target_network({'spatial': non_final_next_spatials,
                                                     'scalar':  non_final_next_scalars})

                invalid_mask = torch.full_like(target_output, -1e9)
                for row_idx, actions_for_state in enumerate(non_final_next_available_actions):
                    valid_actions = list(actions_for_state)
                    if not valid_actions:
                        valid_actions = [_NO_OP]
                    invalid_mask[row_idx, valid_actions] = 0.0

                masked_target_output = target_output + invalid_mask
                next_state_values[non_final_mask] = masked_target_output.max(1)[0]

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