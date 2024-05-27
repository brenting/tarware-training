import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express as px
import torch
import tyro
from gymnasium.spaces import flatten_space
from tarware import RepeatedWarehouse, RewardType

from agent import (
    ActorCritic,
    DirectDistribution,
    HeuristicPolicyModule,
    MultiAgentModule,
    PPOPolicyModule,
)


class PolicyModule(Enum):
    PPO = PPOPolicyModule
    Heuristic = HeuristicPolicyModule

class ModelClass(Enum):
    ActorCritic = ActorCritic
    DirectDistribution = DirectDistribution

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
    observation_type: Literal["flattened", "identifier", "status", "none"] = "flattened"
    normalised_coordinates: bool = False
    render_mode: str | None = None
    action_masking: bool = True
    sample_collection: Literal["all", "masking", "relevant"] = "masking"
    improved_masking: bool = False
    agents_can_clash: bool = True


@dataclass
class Config:
    warehouse: WarehouseConfig

    exp_name: str = Path(__file__).stem
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    wandb: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Warehouse"
    """the wandb's project name"""
    wandb_entity: str | None = "brenting"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Multi-agent settings
    ma_algorithm: Literal["snppo", "ippo"] = "ippo"
    policy_module: Literal[PolicyModule.PPO, PolicyModule.Heuristic] = PolicyModule.PPO
    model_class: Literal[ModelClass.ActorCritic, ModelClass.DirectDistribution] = ModelClass.ActorCritic

    # Algorithm specific arguments
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 1000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.96
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    zero_value: bool = False
    plotting: bool = True


def main():
    config = tyro.cli(Config)
    config.batch_size = config.num_envs * (config.warehouse.n_agvs + config.warehouse.n_pickers) * config.num_steps
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = config.total_timesteps // (config.num_envs * config.num_steps)

    run_name = f"{config.ma_algorithm}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}"
    if config.wandb:
        import wandb
        logger = wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            config=asdict(config),
            name=run_name,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and config.cuda else "cpu")
    config.device = device

    # env setup
    env_config = config.warehouse
    env = RepeatedWarehouse(**asdict(env_config))
    # envs = ss.pettingzoo_env_to_vec_env_v1(env)
    # envs = ss.concat_vec_envs_v1(envs, config.num_envs, num_cpus=config.num_envs)
    # envs.single_observation_space = envs.observation_space
    # envs.single_action_space = envs.action_space
    # envs.is_vector_env = True

    if config.ma_algorithm == "snppo":
        agent_policy_mapping = {agent_id: agent_id.split("_")[0] for agent_id in env.possible_agents}
    elif config.ma_algorithm == "ippo":
        agent_policy_mapping = {agent_id: agent_id for agent_id in env.possible_agents}
    
    agents = MultiAgentModule(agent_policy_mapping, env, config)
    Path(f"models/{run_name}").mkdir(parents=True)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = env.reset(seed=config.seed)
    next_done = {agent_id: False for agent_id in next_obs.keys()}

    actions_data = {agent_id: np.zeros((config.num_iterations // 50, flatten_space(env.action_space(agent_id)).shape[0]), dtype=int) for agent_id in next_obs.keys()}
    actions_relevant_data = {agent_id: np.zeros((config.num_iterations // 50, flatten_space(env.action_space(agent_id)).shape[0]), dtype=int) for agent_id in next_obs.keys()}

    actions_layout_data = {agent_id: np.zeros((config.num_iterations // 50, env.grid_size[0] + 1, env.grid_size[1]), dtype=int) for agent_id in next_obs.keys()}
    actions_layout_relevant_data = {agent_id: np.zeros((config.num_iterations // 50, env.grid_size[0] + 1, env.grid_size[1]), dtype=int) for agent_id in next_obs.keys()}


    for iteration in range(config.num_iterations):
        start_time_iter = time.time()
        # Annealing the rate if instructed to do so.
        if config.anneal_lr:
            frac = 1.0 - iteration / config.num_iterations
            lrnow = frac * config.learning_rate
            for policy in agents.policies.values():
                policy.optimizer.param_groups[0]["lr"] = lrnow

        agents.clear_buffers()

        start_time_rollout = time.time()
        time_counter = defaultdict(float)
        duplicate_action_count = {"AGV": 0, "PICKER": 0}
        log_metrics = defaultdict(list)
        for _ in range(0, config.num_steps):
            global_step += config.num_envs

            agents.add_observations_dones(next_obs, next_done)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                start_tmp = time.time()
                actions = agents.get_actions(next_obs)
                time_counter["policy_sample"] += time.time() - start_tmp
            
            selected_actions = {"AGV": set(), "PICKER": set()}
            for agent_id, action in actions.items():
                action_shifted = action
                agent_type = agent_id.split("_")[0]
                if agent_type == "PICKER" and action != 0:
                    action_shifted = action + len(env.goals)
                    
                location = env.item_loc_dict.get(int(action_shifted), (-1, 0))
                actions_data[agent_id][iteration // 50, action] += 1
                actions_layout_data[agent_id][(iteration // 50,) + location] += 1
                if not infos[agent_id]["busy"]:
                    if action_shifted > (len(env.goals) + 1):
                        if action_shifted in selected_actions[agent_type]:
                            duplicate_action_count[agent_type] += 1
                        selected_actions[agent_type].add(action_shifted)
                    actions_relevant_data[agent_id][iteration // 50, action] += 1
                    actions_layout_relevant_data[agent_id][(iteration // 50,) + location] += 1

            if config.wandb and "__step_common__" in infos:
                logger.log({f"step/{info_id}": data for info_id, data in infos["__step_common__"].items()}, step=global_step, commit=False)


            # TRY NOT TO MODIFY: execute the game and log data.
            start_tmp = time.time()
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            time_counter["step"] += time.time() - start_tmp


            agents.add_rewards(rewards)
            next_done = {agent_id: truncations[agent_id] or term for agent_id, term in terminations.items()}


            for agent_id, done in next_done.items():
                if done:
                    episode_info = infos[agent_id]["episode"]
                    log_metrics[f"mean_episode_return/{agent_id}"].append(float(episode_info["return"]))
                    log_metrics[f"mean_episode_length/{agent_id}"].append(float(episode_info["length"]))

            if "__common__" in infos:
                for metric_name, data in infos["__common__"].items():
                    log_metrics[f"info/{metric_name}"].append(float(data))

        end_time_rollout = time.time()

        log_data = {name: sum(data) / len(data) if len(data) > 0 else 0 for name, data in log_metrics.items()}
        log_data["time/rollout"] = end_time_rollout - start_time_rollout
        log_data["time/step"] = time_counter["step"]
        log_data["time/policy_sample"] = time_counter["policy_sample"]
        for agent_type, count in duplicate_action_count.items():
            log_data[f"info/duplicate_{agent_type}_shelf_action"] = count

        start_time_train = time.time()
        log_data.update(agents.train())
        log_data["time/train"] = time.time() - start_time_train

        if (iteration + 1) % 50 == 0:
            agents.save(f"models/{run_name}")

        log_data["time/iter"] = time.time() - start_time_iter
        log_data["learning_rate"] = list(agents.policies.values())[0].optimizer.param_groups[0]["lr"]
        log_data["SPS"] = int(global_step / (time.time() - start_time))

        if config.wandb:
            if ((iteration + 1) % 50 == 0) and config.plotting:
                log_data.update({f"actions/{agent_id}": px.imshow(actions, text_auto=True) for agent_id, actions in actions_data.items()})
                log_data.update({f"actions_relevant/{agent_id}": px.imshow(actions, text_auto=True) for agent_id, actions in actions_relevant_data.items()})
                log_data.update({f"actions_log/{agent_id}": px.imshow(np.log10(actions + 1), text_auto=True) for agent_id, actions in actions_data.items()})
                log_data.update({f"actions_log_relevant/{agent_id}": px.imshow(np.log10(actions + 1), text_auto=True) for agent_id, actions in actions_relevant_data.items()})
                
                log_data.update({f"actions_layout/{agent_id}": px.imshow(actions[iteration // 50], text_auto=True) for agent_id, actions in actions_layout_data.items()})
                log_data.update({f"actions_layout_relevant/{agent_id}": px.imshow(actions[iteration // 50], text_auto=True) for agent_id, actions in actions_layout_relevant_data.items()})
                log_data.update({f"actions_layout_log/{agent_id}": px.imshow(np.log10(actions[iteration // 50] + 1), text_auto=True) for agent_id, actions in actions_layout_data.items()})
                log_data.update({f"actions_layout_log_relevant/{agent_id}": px.imshow(np.log10(actions[iteration // 50] + 1), text_auto=True) for agent_id, actions in actions_layout_relevant_data.items()})

            log_data["time/iter"] = time.time() - start_time_iter
            logger.log(log_data, step=global_step, commit=True)

        print("SPS:", log_data["SPS"])
        del log_data
    
    env.close()

    if config.wandb:
        # artifact = wandb.Artifact("model", type="model")
        # artifact.add_file(model_path)
        # logger.log_artifact(artifact)
        logger.finish()


if __name__ == "__main__":
    main()
