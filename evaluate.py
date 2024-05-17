import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
import plotly.express as px

import numpy as np
import torch
import tyro
from gymnasium.spaces import flatten_space
from tarware import ObserationType, Warehouse, RewardType

from agent import MultiAgentModule, HeuristicPolicyModule, PPOPolicyModule


@dataclass
class WarehouseConfig:
    column_height: int = 8
    shelf_rows: int = 2
    shelf_columns: int = 5
    n_agvs: int = 8
    n_pickers: int = 4
    msg_bits: int = 0
    sensor_range: int = 1
    request_queue_size: int = 20
    max_inactivity_steps: int | None = None
    max_steps: int = 500
    reward_type: RewardType = RewardType.INDIVIDUAL
    observation_type: ObserationType = ObserationType.FLATTENED
    normalised_coordinates: bool = False
    render_mode: str | None = None
    action_masking: bool = True
    sample_collection: Literal["all", "masking", "relevant"] = "masking"
    no_observations: bool = False
    improved_masking: bool = True


@dataclass
class Config:
    warehouse: WarehouseConfig

    exp_name: str = Path(__file__).stem
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Multi-agent settings
    model: str = "models/ippo_24-05-15_14:24:50"

    num_iterations: int = 10

    learning_rate: float = 3e-4


def main():
    config = tyro.cli(Config)

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and config.cuda else "cpu")
    config.device = device

    # env setup
    env_config = config.warehouse
    env = Warehouse(**asdict(env_config))

    model_type = config.model.split("/")[-1].split("_")[0]
    if model_type == "snppo":
        agent_policy_mapping = {agent_id: agent_id.split("_")[0] for agent_id in env.possible_agents}
    elif model_type == "ippo":
        agent_policy_mapping = {agent_id: agent_id for agent_id in env.possible_agents}
    
    agents = MultiAgentModule(agent_policy_mapping, env, config, module_class=PPOPolicyModule)
    # agents.load(config.model)

    next_obs, infos = env.reset(seed=config.seed)

    actions_layout_data = {agent_id: np.zeros((env.grid_size[0] + 1, env.grid_size[1]), dtype=int) for agent_id in next_obs.keys()}
    actions_layout_relevant_data = {agent_id: np.zeros((env.grid_size[0] + 1, env.grid_size[1]), dtype=int) for agent_id in next_obs.keys()}

    for iteration in range(config.num_iterations):
        print(iteration)
        # agents.clear_buffers()
        next_obs, infos = env.reset(seed=config.seed)
        next_done = {agent_id: False for agent_id in next_obs.keys()}
        episode_done = False
        log_metrics = defaultdict(list)
        while not episode_done:

            # agents.add_observations_dones(next_obs, next_done)
            # ALGO LOGIC: action logic
            with torch.no_grad():
                actions = agents.get_actions(next_obs)
            
            for agent_id, action in actions.items():
                location = env.item_loc_dict.get(int(action), (-1, 0))
                actions_layout_data[agent_id][location] += 1
                if not infos[agent_id]["busy"]:
                    actions_layout_relevant_data[agent_id][location] += 1

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # agents.add_rewards(rewards)
            next_done = {agent_id: truncations[agent_id] or term for agent_id, term in terminations.items()}

            for agent_id, done in next_done.items():
                if done:
                    episode_done = True
                    episode_info = infos[agent_id]["episode"]
                    log_metrics[f"mean_episode_return/{agent_id}"].append(float(episode_info["return"]))
                    log_metrics[f"mean_episode_length/{agent_id}"].append(float(episode_info["length"]))

            if "__common__" in infos:
                for metric_name, data in infos["__common__"].items():
                    log_metrics[f"info/{metric_name}"].append(float(data))


        log_data = {name: sum(data) / len(data) if len(data) > 0 else 0 for name, data in log_metrics.items()}
        # agents.train()

    for name, data in log_data.items():
        print(f"{name}: {data}")
    [px.imshow(actions, text_auto=True).write_html(f"analysis/figures/actions_layout/{agent_id}.html") for agent_id, actions in actions_layout_data.items()]
    [px.imshow(actions, text_auto=True).write_html(f"analysis/figures/actions_layout_relevant/{agent_id}.html") for agent_id, actions in actions_layout_relevant_data.items()]
    [px.imshow(np.log10(actions + 1), text_auto=True).write_html(f"analysis/figures/actions_layout_log/{agent_id}.html") for agent_id, actions in actions_layout_data.items()]
    [px.imshow(np.log10(actions + 1), text_auto=True).write_html(f"analysis/figures/actions_layout_log_relevant/{agent_id}.html") for agent_id, actions in actions_layout_relevant_data.items()]
    [px.imshow(actions, text_auto=True).write_image(f"analysis/figures/actions_layout/{agent_id}.jpg") for agent_id, actions in actions_layout_data.items()]
    [px.imshow(actions, text_auto=True).write_image(f"analysis/figures/actions_layout_relevant/{agent_id}.jpg") for agent_id, actions in actions_layout_relevant_data.items()]
    [px.imshow(np.log10(actions + 1), text_auto=True).write_image(f"analysis/figures/actions_layout_log/{agent_id}.jpg") for agent_id, actions in actions_layout_data.items()]
    [px.imshow(np.log10(actions + 1), text_auto=True).write_image(f"analysis/figures/actions_layout_log_relevant/{agent_id}.jpg") for agent_id, actions in actions_layout_relevant_data.items()]

    env.close()


if __name__ == "__main__":
    main()
