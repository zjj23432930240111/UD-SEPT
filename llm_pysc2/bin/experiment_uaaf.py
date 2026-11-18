# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
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

"""
UAAF experiment entry point for LLM-PySC2.

Configured via environment variables:
- UAAF_TASK: Task number (1, 3, 4, 8)
- UAAF_LEVEL: Difficulty level (1, 2)
- UAAF_STRATEGY: Strategy mode ('raw', 'sc', 'uaaf')
- UAAF_GAME_ID: Game index (0-based)
- UAAF_RESULTS_DIR: Directory to save results
- DEEPSEEK_API_KEY: DeepSeek API key

Usage (via batch runner):
  python -m llm_pysc2.bin.run_uaaf_experiments --num_games 200 --parallel 2

Usage (manual single game):
  set UAAF_TASK=1 && set UAAF_LEVEL=1 && set UAAF_STRATEGY=raw && set UAAF_GAME_ID=0
  python -m pysc2.bin.agent --map pvz_task1_level1 --agent_race protoss --parallel 1
      --agent llm_pysc2.bin.experiment_uaaf.MainAgentUAAF
"""

from llm_pysc2.agents.configs.llm_pysc2 import ConfigPysc2_Harass, ConfigPysc2_Defend, ConfigPysc2_Combat
from llm_pysc2.agents import MainAgent, LLMAgent
from llm_pysc2.lib.uncertainty import ExperimentTracker
import os


def get_config(task):
    """Get task-specific config, same logic as experiment_llm_pysc2.py."""
    if task in [1, 2]:
        config = ConfigPysc2_Harass()
        for agent_name in list(config.AGENTS.keys()):
            for team in config.AGENTS[agent_name]['team']:
                team['task'] = [
                    {'time': None, 'pos': [52, 32],
                     'info': "Go to minimap coordinate [52, 32], and try to avoid been detected or attacked before arrival."},
                    {'time': None, 'pos': None,
                     'info': "Kill as much as enemy workers as possible."},
                ]
    elif task in [3]:
        config = ConfigPysc2_Defend()
        for team in config.AGENTS['CombatGroup1']['team']:
            team['task'] = [
                {'time': '0:00', 'pos': None,
                 'info': "Protect our nexus and probes from enemy airdrops. At Game time 0:00, "
                         "2 airdrops detected from minimap [24, 32] and [12, 24] to [16, 32]"},
                {'time': '0:10', 'pos': None,
                 'info': "Protect our nexus and probes from enemy airdrops. At Game time 0:10, "
                         "2 airdrops detected from minimap [20, 24] and [20, 40] to [16, 32]"},
                {'time': '0:20', 'pos': None,
                 'info': "Protect our nexus and probes from enemy airdrops. At Game time 0:20, "
                         "2 airdrops detected from minimap [24, 32] and [12, 40] to [16, 32]"},
                {'time': '0:30', 'pos': None,
                 'info': "Protect our nexus and probes from enemy airdrops. At Game time 0:30, "
                         "2 airdrops detected from minimap [24, 32] and [10, 32] to [16, 32]"},
            ]
    elif task in [4, 5, 6]:
        config = ConfigPysc2_Combat()
        for agent_name in list(config.AGENTS.keys()):
            for team in config.AGENTS[agent_name]['team']:
                team['task'] = [
                    {'time': None, 'pos': [32, 32],
                     'info': "Go to minimap coordinate [32, 32]."},
                    {'time': '0:10', 'pos': None,
                     'info': "Kill as much as enemy units as possible. If no enemy found, hold the position."},
                ]
    elif task in [8]:
        config = ConfigPysc2_Combat()
        config.ENABLE_COMMUNICATION = True
        config.AGENTS_ALWAYS_DISABLE.remove('Airborne')
        config.AGENTS_ALWAYS_DISABLE.remove('Commander')
        config.AGENTS_ALWAYS_DISABLE.remove('Developer')
        config.AGENTS['Commander']['team'][0]['task'] = [
            {'time': None, 'pos': None,
             'info': "Organize a multiline combat to defeat enemy troops and kill their workers, "
                     "you should reach all the goals and finish the battle before game time 1:30."},
        ]
    else:
        raise AssertionError(f"Wrong task index: {task}")

    return config


def setup_config(task, level, strategy, api_key):
    """Set up config with DeepSeek LLM and UAAF strategy parameters."""
    config = get_config(task)

    # Configure DeepSeek LLM
    # SC/UAAF need higher temperature for diversity; Raw uses low temperature
    temperature = 0.5 if strategy in ('sc', 'uaaf') else 0.1
    config.reset_llm(
        model_name='deepseek-v4-pro',
        api_base='https://api.deepseek.com/v1',
        api_key=api_key,
        temperature=temperature
    )

    # Configure UAAF strategy parameters
    config.strategy_mode = strategy
    config.num_sc_calls = 3
    config.num_uaaf_calls = 4
    config.entropy_threshold = 1.0
    config.sc_temperature = 0.5

    # Configure fallback action by task type
    if task in [1, 2]:
        config.fallback_action_name = 'Stop'  # Harass tasks: stop moving
    else:
        config.fallback_action_name = 'Hold_Position'  # Combat/defend tasks: hold position

    # Bind experiment tracker
    results_dir = os.environ.get('UAAF_RESULTS_DIR', 'uaaf_results')
    game_id = int(os.environ.get('UAAF_GAME_ID', '0'))
    tracker = ExperimentTracker(results_dir, task, level, strategy, game_id)
    config.experiment_tracker = tracker

    return config


class MainAgentUAAF(MainAgent):
    """MainAgent subclass for UAAF experiments.

    Reads configuration from environment variables set by the batch runner.
    """

    def __init__(self):
        task = int(os.environ.get('UAAF_TASK', '1'))
        level = int(os.environ.get('UAAF_LEVEL', '1'))
        strategy = os.environ.get('UAAF_STRATEGY', 'raw')
        api_key = os.environ.get('DEEPSEEK_API_KEY', 'sk-764a21b2e23d4b848bbc7347b2f413db')

        config = setup_config(task, level, strategy, api_key)
        super(MainAgentUAAF, self).__init__(config, LLMAgent)

    def step(self, obs):
        return super().step(obs)


if __name__ == "__main__":
    level = int(os.environ.get('UAAF_LEVEL', '1'))
    task = int(os.environ.get('UAAF_TASK', '1'))
    map_name = f"pvz_task{task}_level{level}"
    os.system(f"python -m pysc2.bin.agent --map {map_name} --agent_race protoss --parallel 1 "
              f"--agent llm_pysc2.bin.experiment_uaaf.MainAgentUAAF")
