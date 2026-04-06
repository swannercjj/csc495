#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

import importlib
import inspect
import threading
import run_loop
import os
import tempfile
from datetime import datetime

from absl import app
from absl import flags

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from agent.artifact_store import is_s3_uri, join_artifact_path, upload_file

try:
    import wandb
except ImportError:
    wandb = None


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", False,
                  "Whether to include raw units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_enum("artifact_backend", "local", ["local", "s3"],
                  "Where to store checkpoints and replays.")
flags.DEFINE_string("checkpoint_dir", "checkpoints",
                    "Directory or s3:// prefix for training checkpoints.")
flags.DEFINE_string("replay_dir", "replays",
                    "Directory or s3:// prefix for saved replays.")
flags.DEFINE_string("resume_checkpoint", None,
                    "Optional checkpoint path or s3:// URI to resume from.")
flags.DEFINE_string("sc2path", None,
                    "Optional StarCraft II install path for local or remote runs.")
flags.DEFINE_bool("headless", False,
                    "Force headless execution even if render is enabled.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.mark_flag_as_required("map")

flags.DEFINE_bool("wandb", False, "Enable Weights & Biases logging.")
flags.DEFINE_string("wandb_entity", "swannercjj", "W&B entity.")
flags.DEFINE_string("wandb_project", "csc495", "W&B project name.")
flags.DEFINE_string("wandb_run_name", None, "Optional W&B run name.")


def _instantiate_agent(agent_cls, checkpoint_dir=None, resume_checkpoint=None):
    kwargs = {}
    signature = inspect.signature(agent_cls)
    if "checkpoint_dir" in signature.parameters:
        kwargs["checkpoint_dir"] = checkpoint_dir
    if "resume_checkpoint" in signature.parameters and resume_checkpoint:
        kwargs["resume_checkpoint"] = resume_checkpoint
    return agent_cls(**kwargs)


def run_thread(agent_classes, players, map_name, visualize, checkpoint_dir=None,
               replay_dir=None, resume_checkpoint=None):
    """Run one thread worth of the environment with agents."""
    with sc2_env.SC2Env(
            map_name=map_name,
            battle_net_map=FLAGS.battle_net_map,
            players=players,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=FLAGS.feature_screen_size,
                feature_minimap=FLAGS.feature_minimap_size,
                rgb_screen=FLAGS.rgb_screen_size,
                rgb_minimap=FLAGS.rgb_minimap_size,
                action_space=FLAGS.action_space,
                use_feature_units=FLAGS.use_feature_units,
                use_raw_units=FLAGS.use_raw_units),
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog,
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        if checkpoint_dir is not None:
            agents = []
            for index, agent_cls in enumerate(agent_classes):
                agents.append(_instantiate_agent(
                    agent_cls,
                    checkpoint_dir=checkpoint_dir,
                    resume_checkpoint=resume_checkpoint if index == 0 else None,
                ))
        else:
            agents = []
            for index, agent_cls in enumerate(agent_classes):
                agents.append(_instantiate_agent(
                    agent_cls,
                    resume_checkpoint=resume_checkpoint if index == 0 else None,
                ))
        run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes, checkpoint_dir=checkpoint_dir)
        if FLAGS.save_replay:
            env.save_replay(agent_classes[0].__name__)
            # local_replay_dir = replay_dir
            # if FLAGS.artifact_backend == "s3" and replay_dir and is_s3_uri(replay_dir):
            #     local_replay_dir = tempfile.mkdtemp(prefix="csc495_replays_")

            # replay_path = env.save_replay(local_replay_dir or os.getcwd(), agent_classes[0].__name__)
            # if FLAGS.artifact_backend == "s3" and replay_dir and is_s3_uri(replay_dir):
            #     remote_replay_path = join_artifact_path(replay_dir, os.path.basename(replay_path))
            #     upload_file(replay_path, remote_replay_path)


def main(args):
    """Run an agent."""
    if FLAGS.sc2path:
        os.environ["SC2PATH"] = FLAGS.sc2path

    if FLAGS.trace:
        stopwatch.sw.trace()
    elif FLAGS.profile:
        stopwatch.sw.enable()

    if wandb is not None and FLAGS.wandb:
        wandb.init(
            entity=FLAGS.wandb_entity,
            project=FLAGS.wandb_project,
            name=FLAGS.wandb_run_name,
            config={
                "map": FLAGS.map,
                "agent": FLAGS.agent,
                "agent2": FLAGS.agent2,
                "difficulty": FLAGS.difficulty,
                "step_mul": FLAGS.step_mul,
                "max_episodes": FLAGS.max_episodes,
                "max_agent_steps": FLAGS.max_agent_steps,
            },
        )

    map_inst = maps.get(FLAGS.map)

    agent_classes = []
    players = []

    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_classes.append(agent_cls)
    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                                FLAGS.agent_name or agent_name))

    if map_inst.players >= 2:
        if FLAGS.agent2 == "Bot":
            players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                        sc2_env.Difficulty[FLAGS.difficulty],
                                        sc2_env.BotBuild[FLAGS.bot_build]))
        else:
            agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
            agent_cls = getattr(importlib.import_module(agent_module), agent_name)
            agent_classes.append(agent_cls)
            players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race],
                                        FLAGS.agent2_name or agent_name))

    # Build a per-run checkpoint directory based on agent type and time.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    primary_agent_name = agent_classes[0].__name__
    checkpoint_dir = join_artifact_path(FLAGS.checkpoint_dir, primary_agent_name, run_timestamp)
    replay_dir = join_artifact_path(FLAGS.replay_dir, primary_agent_name, run_timestamp)
    resume_checkpoint = FLAGS.resume_checkpoint

    if FLAGS.artifact_backend == "local":
        if is_s3_uri(FLAGS.checkpoint_dir) or is_s3_uri(FLAGS.replay_dir) or is_s3_uri(FLAGS.resume_checkpoint):
            raise ValueError("artifact_backend=local cannot be used with s3:// paths")

    threads = []
    for _ in range(FLAGS.parallel - 1):
        t = threading.Thread(target=run_thread,
                            args=(agent_classes, players, FLAGS.map, False,
                                  checkpoint_dir, replay_dir, resume_checkpoint))
        threads.append(t)
        t.start()

    run_thread(agent_classes, players, FLAGS.map,
               FLAGS.render and not FLAGS.headless,
               checkpoint_dir, replay_dir, resume_checkpoint)

    for t in threads:
        t.join()

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)