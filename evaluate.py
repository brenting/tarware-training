import random
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import plotly.express as px
import torch
import tyro
from tarware import Warehouse

from agent import MultiAgentModule
from train import Config


def main():
    config = tyro.cli(Config)
    config.num_iterations = 1
    config.model = "models/ippo_24-05-24_19:42:38"

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

    if config.ma_algorithm == "snppo":
        agent_policy_mapping = {agent_id: agent_id.split("_")[0] for agent_id in env.possible_agents}
    elif config.ma_algorithm == "ippo":
        agent_policy_mapping = {agent_id: agent_id for agent_id in env.possible_agents}
    
    agents = MultiAgentModule(agent_policy_mapping, env, config)
    agents.load(config.model)

    next_obs, infos = env.reset(seed=config.seed)

    actions_layout_data = {agent_id: np.zeros((env.grid_size[0] + 1, env.grid_size[1]), dtype=int) for agent_id in next_obs.keys()}
    actions_layout_relevant_data = {agent_id: np.zeros((env.grid_size[0] + 1, env.grid_size[1]), dtype=int) for agent_id in next_obs.keys()}

    log_metrics = defaultdict(list)
    for iteration in range(config.num_iterations):
        print(iteration)
        # agents.clear_buffers()
        # env = Warehouse(**asdict(env_config))
        # agents = MultiAgentModule(agent_policy_mapping, env, config)
        next_obs, infos = env.reset(seed=config.seed + iteration)
        next_done = {agent_id: False for agent_id in next_obs.keys()}
        episode_done = False
        while not episode_done:

            env.render()

            # agents.add_observations_dones(next_obs, next_done)
            # ALGO LOGIC: action logic
            with torch.no_grad():
                actions = agents.get_actions(next_obs)
            
            for agent_id, action in actions.items():
                action_shifted = action + len(env.goals) if agent_id.startswith("PICKER") and action != 0 else action
                location = env.item_loc_dict.get(int(action_shifted), (-1, 0))
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

        print(f"pickrate: {sum(log_metrics['info/pickrate']) / len(log_metrics['info/pickrate'])}")

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
