import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from tarware import ObserationType, RepeatedWarehouse, RewardType

from agent import MultiAgentModule


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
    wandb: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Warehouse"
    """the wandb's project name"""
    wandb_entity: str | None = "brenting"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Multi-agent settings
    ma_algorithm: Literal["snppo", "ippo"] = "snppo"

    # Algorithm specific arguments
    total_timesteps: int = 4_000_000
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
    num_minibatches: int = 30
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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

    if config.ma_algorithm == "ippo":
        agent_policy_mapping = {agent_id: agent_id.split("_")[0] for agent_id in env.possible_agents}
    elif config.ma_algorithm == "snppo":
        agent_policy_mapping = {agent_id: agent_id for agent_id in env.possible_agents}
    
    agents = MultiAgentModule(agent_policy_mapping, env, config)
    Path(f"models/{run_name}").mkdir(parents=True)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = env.reset(seed=config.seed)
    next_done = {agent_id: False for agent_id in next_obs.keys()}

    for iteration in range(1, config.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.learning_rate
            for policy in agents.policies.values():
                policy.optimizer.param_groups[0]["lr"] = lrnow

        agents.clear_buffers()

        log_metrics = defaultdict(list)
        for _ in range(0, config.num_steps + 1):
            global_step += config.num_envs

            agents.add_to_buffer({"dones": next_done})

            # ALGO LOGIC: action logic
            with torch.no_grad():
                actions = agents.get_actions(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            agents.add_to_buffer({"rewards": rewards})
            next_done = {agent_id: truncations[agent_id] or term for agent_id, term in terminations.items()}

            # next_done_np = np.logical_or(terminations, truncations)
            # buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs = torch.Tensor(next_obs).to(device)
            # next_done = torch.Tensor(next_done_np).to(device)

            if any(list(next_done.values())):
                for agent_id, done in next_done.items():
                    if done:
                        episode_info = infos[agent_id]["episode"]
                        log_metrics["mean_episode_return"].append(float(episode_info["return"]))
                        log_metrics["mean_episode_length"].append(float(episode_info["length"]))

            if "__common__" in infos:
                for metric_name, data in infos["__common__"].items():
                    log_metrics[metric_name].append(float(data))
        

        for name, data in log_metrics.items():
            mean = sum(data) / len(data) if len(data) > 0 else 0
            if config.wandb:
                logger.log({name: mean}, step=global_step, commit=False)
            print(f"{name}: {mean}")

        log_data = agents.train()

        agents.save(f"models/{run_name}")

        if config.wandb:
            log_data.update({
                "learning_rate": list(agents.policies.values())[0].optimizer.param_groups[0]["lr"],
                "measures/SPS": int(global_step / (time.time() - start_time)),
            })
            logger.log(log_data, step=global_step, commit=True)

        print("SPS:", int(global_step / (time.time() - start_time)))
    
    env.close()

    if config.wandb:
        # artifact = wandb.Artifact("model", type="model")
        # artifact.add_file(model_path)
        # logger.log_artifact(artifact)
        logger.finish()


if __name__ == "__main__":
    main()
