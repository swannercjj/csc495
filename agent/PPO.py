import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical

from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
from pysc2.lib import features

from agent.alphastar.encoders import SpatialEncoder, ScalarEncoder
from agent.artifact_store import load_torch_state, save_torch_state

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

_NO_OP = actions.FUNCTIONS.no_op.id
_PLAYER_SELF = features.PlayerRelative.SELF

SCALAR_HIDDEN_DIM = 128
SPATIAL_HIDDEN_DIM = 256
MAX_SPATIAL_POINTS = 2


class PPOObservationPreprocessor:
    """Build model inputs from raw PySC2 observations for PPO."""

    def __init__(self, device: torch.device):
        self.device = device

    def preprocess(self, timestep):
        observation = timestep.observation
        screen = observation["feature_screen"]

        planes = np.stack(
            [
                screen.player_relative,
                screen.unit_type,
                screen.selected,
                screen.unit_hit_points_ratio,
                screen.unit_shields_ratio,
                screen.unit_density,
                screen.creep,
                screen.height_map,
            ],
            axis=0,
        ).astype(np.float32)

        player_data = np.asarray(observation.player, dtype=np.float32)
        player_data = np.nan_to_num(
            player_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )

        return {
            "spatial": torch.from_numpy(planes).unsqueeze(0).to(self.device),
            "scalar": torch.from_numpy(player_data).unsqueeze(0).to(self.device),
        }


def _is_spatial_arg(arg_spec) -> bool:
    sizes = getattr(arg_spec, "sizes", ())
    return len(sizes) == 2 and all(size > 1 for size in sizes)


class PPOActorCritic(nn.Module):
    """Factorized actor-critic network for PySC2 function and argument selection."""

    def __init__(self, n_actions: int, arg_types: list):
        super().__init__()
        self.spatial_encoder = SpatialEncoder()
        self.scalar_encoder = ScalarEncoder()

        combined_dim = SPATIAL_HIDDEN_DIM + SCALAR_HIDDEN_DIM

        self.function_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        self.spatial_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.arg_heads = nn.ModuleDict()
        for arg in arg_types:
            sizes = getattr(arg, "sizes", ())
            if len(sizes) != 1:
                continue
            output_dim = int(sizes[0])
            if output_dim <= 1:
                continue
            self.arg_heads[arg.name] = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

    def forward(self, obs_dict: dict) -> dict:
        map_skip, spatial_enc = self.spatial_encoder(obs_dict["spatial"])
        scalar_enc = self.scalar_encoder(obs_dict["scalar"])
        combined = torch.cat([spatial_enc, scalar_enc], dim=1)

        function_logits = self.function_head(combined)
        spatial_logits = self.spatial_head(map_skip).squeeze(1)

        target_h, target_w = obs_dict["spatial"].shape[-2:]
        if spatial_logits.shape[-2:] != (target_h, target_w):
            spatial_logits = F.interpolate(
                spatial_logits.unsqueeze(1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        value = self.value_head(combined).squeeze(1)

        return {
            "function_logits": function_logits,
            "spatial_logits": spatial_logits,
            "arg_logits": {
                name: head(combined) for name, head in self.arg_heads.items()
            },
            "value": value,
        }


class PPOAgent(BaseAgent):
    def __init__(
        self,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        batch_size=64,
        ppo_epochs=4,
        rollout_steps=512,
        checkpoint_interval=100,
        checkpoint_dir=None,
        resume_checkpoint=None,
    ):
        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.rollout_steps = rollout_steps

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.resume_checkpoint = resume_checkpoint

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_preprocessor = PPOObservationPreprocessor(self.device)

        self.steps_done = 0
        self.episode_index = 0
        self.episode_step = 0
        self.episode_return = 0.0
        self.raw_episode_return = 0.0
        self.episode_score = 0.0
        self.prev_self_health = 0.0

        self.last_state = None
        self.last_action = None
        self.last_available_actions = None
        self.last_log_prob = None
        self.last_value = None

        self.last_policy_loss = None
        self.last_value_loss = None

        self.rollout = []

    def save_checkpoint(self, episode_index=None, checkpoint_dir=None):
        base_dir = checkpoint_dir or self.checkpoint_dir
        if base_dir is None:
            base_dir = "checkpoints"

        if episode_index is None:
            episode_index = self.episode_index

        filename = f"ppo_ep_{episode_index}.pt"
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode_index,
            "steps_done": self.steps_done,
        }
        save_torch_state(state, base_dir, filename)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = load_torch_state(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_index = checkpoint.get("episode", 0)
        self.steps_done = checkpoint.get("steps_done", 0)

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

        self.n_actions = len(action_spec.functions)
        self.arg_types = sorted(actions.TYPES._asdict().values(), key=lambda a: a.id)
        self.num_arg_types = max(arg.id for arg in self.arg_types) + 1

        self.model = PPOActorCritic(self.n_actions, self.arg_types).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.resume_checkpoint:
            self.load_checkpoint(self.resume_checkpoint)

    def reset(self):
        super().reset()
        self.prev_score = np.zeros(13, dtype=np.float32)
        self.prev_self_health = 0.0

        self.last_state = None
        self.last_action = None
        self.last_available_actions = None
        self.last_log_prob = None
        self.last_value = None

        self.episode_index += 1
        self.episode_step = 0
        self.episode_return = 0.0
        self.raw_episode_return = 0.0
        self.episode_score = 0.0

    def _get_obs(self, obs):
        return self.obs_preprocessor.preprocess(obs)

    def _encode_action(self, function_id, spatial_points, arg_value_map):
        packed = np.full(1 + (MAX_SPATIAL_POINTS * 2) + self.num_arg_types, -1, dtype=np.int64)
        packed[0] = int(function_id)

        if len(spatial_points) > 0:
            packed[1] = int(spatial_points[0][0])
            packed[2] = int(spatial_points[0][1])
        if len(spatial_points) > 1:
            packed[3] = int(spatial_points[1][0])
            packed[4] = int(spatial_points[1][1])

        for arg_id, arg_value in arg_value_map.items():
            packed[1 + (MAX_SPATIAL_POINTS * 2) + int(arg_id)] = int(arg_value)

        return torch.tensor([packed], device=self.device, dtype=torch.long)

    def _project_q_point_to_arg(self, q_point, arg_spec, q_width, q_height):
        arg_width = max(1, int(arg_spec.sizes[0]))
        arg_height = max(1, int(arg_spec.sizes[1]))

        x_q = int(np.clip(q_point[0], 0, q_width - 1))
        y_q = int(np.clip(q_point[1], 0, q_height - 1))

        if q_width == 1:
            x_arg = 0
        else:
            x_arg = int(round(x_q * (arg_width - 1) / (q_width - 1)))

        if q_height == 1:
            y_arg = 0
        else:
            y_arg = int(round(y_q * (arg_height - 1) / (q_height - 1)))

        return [x_arg, y_arg]

    def _action_to_function_call(self, function_id, spatial_points, arg_value_map, q_map_shape):
        func = self.action_spec.functions[function_id]
        args = []
        spatial_idx = 0

        for arg in func.args:
            if _is_spatial_arg(arg):
                if spatial_idx >= len(spatial_points):
                    raise ValueError("Missing spatial point for action.")
                point = self._project_q_point_to_arg(
                    spatial_points[spatial_idx],
                    arg,
                    q_width=q_map_shape[0],
                    q_height=q_map_shape[1],
                )
                args.append(point)
                spatial_idx += 1
            else:
                value = arg_value_map.get(arg.id)
                if value is None:
                    raise ValueError(f"Missing non-spatial arg for id {arg.id}.")
                value = int(np.clip(value, 0, int(arg.sizes[0]) - 1))
                args.append([value])

        return actions.FunctionCall(function_id, args)

    def _sample_action(self, state, available_actions):
        output = self.model(state)
        function_logits = output["function_logits"]
        spatial_logits = output["spatial_logits"]
        arg_logits = output["arg_logits"]

        mask = torch.full((self.n_actions,), -1e9, device=self.device)
        valid_actions = list(available_actions)
        if len(valid_actions) == 0:
            valid_actions = [_NO_OP]
        mask[valid_actions] = 0.0

        masked_function_logits = function_logits[0] + mask
        function_dist = Categorical(logits=masked_function_logits)
        function_id = int(function_dist.sample().item())
        log_prob = function_dist.log_prob(torch.tensor(function_id, device=self.device))
        entropy = function_dist.entropy()

        spatial_q_map = spatial_logits[0]
        height, width = spatial_q_map.shape

        chosen_spatial_points = []
        chosen_arg_values = {}

        func = self.action_spec.functions[function_id]
        flat_spatial_logits = spatial_q_map.reshape(-1)
        spatial_dist = Categorical(logits=flat_spatial_logits)

        for arg in func.args:
            if _is_spatial_arg(arg):
                flat_idx = int(spatial_dist.sample().item())
                y = flat_idx // width
                x = flat_idx % width
                chosen_spatial_points.append([x, y])
                log_prob = log_prob + spatial_dist.log_prob(torch.tensor(flat_idx, device=self.device))
                entropy = entropy + spatial_dist.entropy()
            else:
                logits = arg_logits.get(arg.name)
                if logits is None:
                    raise ValueError(f"Missing policy head for arg type: {arg.name}")
                arg_dist = Categorical(logits=logits[0])
                arg_value = int(arg_dist.sample().item())
                chosen_arg_values[arg.id] = arg_value
                log_prob = log_prob + arg_dist.log_prob(torch.tensor(arg_value, device=self.device))
                entropy = entropy + arg_dist.entropy()

        action_tensor = self._encode_action(function_id, chosen_spatial_points, chosen_arg_values)
        return (
            function_id,
            chosen_spatial_points,
            chosen_arg_values,
            action_tensor,
            log_prob.detach(),
            output["value"][0].detach(),
            entropy.detach(),
        )

    def _evaluate_actions(self, states, actions_batch, available_actions_list):
        output = self.model(states)
        function_logits = output["function_logits"]
        spatial_logits = output["spatial_logits"]
        arg_logits = output["arg_logits"]
        values = output["value"]

        function_ids = actions_batch[:, 0].long()
        point1_x = actions_batch[:, 1].long()
        point1_y = actions_batch[:, 2].long()
        point2_x = actions_batch[:, 3].long()
        point2_y = actions_batch[:, 4].long()
        arg_action_batch = actions_batch[:, 1 + (MAX_SPATIAL_POINTS * 2):].long()

        batch_size = function_ids.shape[0]
        log_probs = torch.zeros(batch_size, device=self.device)
        entropies = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            available_actions = available_actions_list[i]
            valid_actions = list(available_actions)
            if len(valid_actions) == 0:
                valid_actions = [_NO_OP]

            mask = torch.full((self.n_actions,), -1e9, device=self.device)
            mask[valid_actions] = 0.0

            f_dist = Categorical(logits=function_logits[i] + mask)
            function_id = int(function_ids[i].item())
            log_probs[i] = log_probs[i] + f_dist.log_prob(torch.tensor(function_id, device=self.device))
            entropies[i] = entropies[i] + f_dist.entropy()

            flat_spatial_logits = spatial_logits[i].reshape(-1)
            width = spatial_logits.shape[2]
            s_dist = Categorical(logits=flat_spatial_logits)

            if point1_x[i] >= 0 and point1_y[i] >= 0:
                idx1 = int(point1_y[i].item() * width + point1_x[i].item())
                log_probs[i] = log_probs[i] + s_dist.log_prob(torch.tensor(idx1, device=self.device))
                entropies[i] = entropies[i] + s_dist.entropy()

            if point2_x[i] >= 0 and point2_y[i] >= 0:
                idx2 = int(point2_y[i].item() * width + point2_x[i].item())
                log_probs[i] = log_probs[i] + s_dist.log_prob(torch.tensor(idx2, device=self.device))
                entropies[i] = entropies[i] + s_dist.entropy()

            func = self.action_spec.functions[function_id]
            for arg in func.args:
                if _is_spatial_arg(arg):
                    continue
                if arg.name not in arg_logits:
                    continue
                arg_idx = int(arg.id)
                if arg_idx >= arg_action_batch.shape[1]:
                    continue
                arg_val = int(arg_action_batch[i, arg_idx].item())
                if arg_val < 0:
                    continue

                a_dist = Categorical(logits=arg_logits[arg.name][i])
                log_probs[i] = log_probs[i] + a_dist.log_prob(torch.tensor(arg_val, device=self.device))
                entropies[i] = entropies[i] + a_dist.entropy()

        return log_probs, entropies, values

    def _compute_returns_and_advantages(self, rewards, dones, values, bootstrap_value):
        n = rewards.shape[0]
        advantages = torch.zeros(n, device=self.device)
        returns = torch.zeros(n, device=self.device)

        gae = 0.0
        next_value = bootstrap_value

        for t in reversed(range(n)):
            non_terminal = 1.0 - float(dones[t].item())
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False) + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return returns, advantages

    def _optimize(self, bootstrap_value=0.0):
        if len(self.rollout) == 0:
            return

        state_spatials = torch.cat([item["state"]["spatial"] for item in self.rollout], dim=0)
        state_scalars = torch.cat([item["state"]["scalar"] for item in self.rollout], dim=0)

        states = {"spatial": state_spatials, "scalar": state_scalars}
        actions_batch = torch.cat([item["action"] for item in self.rollout], dim=0)

        old_log_probs = torch.stack([item["log_prob"] for item in self.rollout]).detach()
        old_values = torch.stack([item["value"] for item in self.rollout]).detach()
        rewards = torch.tensor([item["reward"] for item in self.rollout], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item["done"] for item in self.rollout], dtype=torch.float32, device=self.device)
        available_actions_list = [item["available_actions"] for item in self.rollout]

        bootstrap = torch.tensor(float(bootstrap_value), dtype=torch.float32, device=self.device)
        returns, advantages = self._compute_returns_and_advantages(
            rewards=rewards,
            dones=dones,
            values=old_values,
            bootstrap_value=bootstrap,
        )

        n = actions_batch.shape[0]
        indices = np.arange(n)

        last_policy_loss = 0.0
        last_value_loss = 0.0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                mb_idx = indices[start:end]

                mb_states = {
                    "spatial": states["spatial"][mb_idx],
                    "scalar": states["scalar"][mb_idx],
                }
                mb_actions = actions_batch[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_available_actions = [available_actions_list[i] for i in mb_idx]

                new_log_probs, entropies, values = self._evaluate_actions(
                    mb_states,
                    mb_actions,
                    mb_available_actions,
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                entropy_bonus = entropies.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())

        self.last_policy_loss = last_policy_loss
        self.last_value_loss = last_value_loss
        self.rollout = []

    def _get_shaped_reward(self, obs):
        current_score = np.asarray(obs.observation["score_cumulative"], dtype=np.float32)
        current_self_health = self._get_total_self_health(obs)

        if self.episode_step == 0:
            self.prev_score = current_score
            self.prev_self_health = current_self_health
            return 0.0

        delta = current_score - self.prev_score
        self.prev_score = current_score

        health_delta = current_self_health - self.prev_self_health
        self.prev_self_health = current_self_health

        weights = {
            "mineral_collected": 0.005,
            "vespene_collected": 0.005,
            "kill_unit": 0.1,
            "kill_building": 0.2,
            "spent_resources": 0.005,
            "health_loss": 0.02,
            "time_penalty": -0.0005,
            "win_loss": 1.0,
        }

        spent_delta = delta[11] + delta[12]
        health_loss_penalty = min(0.0, health_delta) * weights["health_loss"]

        total_reward = (
            (delta[7] * weights["mineral_collected"])
            + (delta[8] * weights["vespene_collected"])
            + (delta[5] * weights["kill_unit"])
            + (delta[6] * weights["kill_building"])
            + (spent_delta * weights["spent_resources"])
            + health_loss_penalty
            + weights["time_penalty"]
            + (obs.reward * weights["win_loss"])
        )

        return float(np.clip(total_reward, -1.0, 1.0))

    def _get_total_self_health(self, obs):
        screen = obs.observation["feature_screen"]
        if screen is None:
            return 0.0

        player_relative = np.asarray(screen.player_relative)
        hp_ratio = np.asarray(screen.unit_hit_points_ratio, dtype=np.float32)
        shield_ratio = np.asarray(screen.unit_shields_ratio, dtype=np.float32)

        if hp_ratio.max(initial=0.0) > 1.0:
            hp_ratio = hp_ratio / 255.0
        if shield_ratio.max(initial=0.0) > 1.0:
            shield_ratio = shield_ratio / 255.0

        self_mask = player_relative == _PLAYER_SELF
        total_health = (hp_ratio + shield_ratio)[self_mask].sum()
        return float(total_health)

    def step(self, obs):
        super().step(obs)

        step_reward = self._get_shaped_reward(obs)
        raw_step_reward = float(obs.reward)
        done = obs.last()

        self.episode_return += step_reward
        self.raw_episode_return += raw_step_reward
        self.episode_step += 1
        self.steps_done += 1

        state = self._get_obs(obs)
        available_actions = obs.observation["available_actions"]
        if len(available_actions) == 0:
            available_actions = [_NO_OP]

        if self.last_state is not None and self.last_action is not None:
            self.rollout.append(
                {
                    "state": self.last_state,
                    "action": self.last_action,
                    "available_actions": self.last_available_actions,
                    "log_prob": self.last_log_prob,
                    "value": self.last_value,
                    "reward": step_reward,
                    "done": float(done),
                }
            )

        if done:
            self._optimize(bootstrap_value=0.0)

            current_score = np.asarray(obs.observation["score_cumulative"], dtype=np.float32)
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
                        "ppo_policy_loss": float(self.last_policy_loss) if self.last_policy_loss is not None else 0.0,
                        "ppo_value_loss": float(self.last_value_loss) if self.last_value_loss is not None else 0.0,
                        "outcome": raw_step_reward,
                    }
                )

            self.last_state = None
            self.last_action = None
            self.last_available_actions = None
            self.last_log_prob = None
            self.last_value = None

            return actions.FunctionCall(_NO_OP, [])

        (
            function_id,
            chosen_spatial_points,
            chosen_arg_values,
            action_tensor,
            log_prob,
            value,
            _entropy,
        ) = self._sample_action(state, available_actions)

        spatial_height, spatial_width = state["spatial"].shape[-2:]
        function_call = self._action_to_function_call(
            function_id=function_id,
            spatial_points=chosen_spatial_points,
            arg_value_map=chosen_arg_values,
            q_map_shape=(spatial_width, spatial_height),
        )

        self.last_state = state
        self.last_action = action_tensor
        self.last_available_actions = available_actions
        self.last_log_prob = log_prob
        self.last_value = value

        if len(self.rollout) >= self.rollout_steps:
            with torch.no_grad():
                bootstrap_value = self.model(state)["value"][0].item()
            self._optimize(bootstrap_value=bootstrap_value)

        return function_call
