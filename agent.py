from collections import defaultdict
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import flatten_space
from numpy import ndarray
from pettingzoo import ParallelEnv
from tensordict import TensorDict
from torch import Tensor, nn, optim
from torch.distributions import Categorical


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.init_buffer()

    def init_buffer(self):
        self.has_samples = False

        self.observations = []
        self.dones = []
        self.actions = []
        self.values = []
        self.logprobs = []
        self.rewards = []

        self.action_masks = []
        self.sample_masks = []

    def add_observation_done(self, observation: dict[str, ndarray], done: bool):
        self.observation = observation
        self.done = done

    def add_policy_sample(self, action: Tensor, value: Tensor, logprob: Tensor):
        self.action = action
        self.value = value
        self.logprob = logprob

    def add_reward(self, reward: np.float32):
        self.reward = reward
        self.flush_full_sample()

    def flush_full_sample(self):
        self.has_samples = True

        self.observations.append(self.observation["observation"])
        self.dones.append(self.done)
        self.actions.append(self.action)
        self.values.append(self.value)
        self.logprobs.append(self.logprob)
        self.rewards.append(self.reward)

        if "action_mask" in self.observation:
            self.action_masks.append(self.observation["action_mask"])
        if "sample_mask" in self.observation:
            self.sample_masks.append(self.observation["sample_mask"])

    def get_train_batch(self, policy, config) -> TensorDict:
        batch_size = len(self.observations)

        batch = {
            "observations": torch.tensor(np.array(self.observations, dtype=np.float32).squeeze(), device=config.device),
            "dones": torch.tensor(np.array(self.dones, dtype=np.float32).squeeze(), device=config.device),
            "rewards": torch.tensor(np.array(self.rewards, dtype=np.float32).squeeze(), device=config.device),
            "actions": torch.tensor(self.actions, dtype=torch.int64, device=config.device).squeeze(),
            "values": torch.tensor(self.values, dtype=torch.float32, device=config.device).squeeze(),
            "logprobs": torch.tensor(self.logprobs, dtype=torch.float32, device=config.device).squeeze(),
        }
        if self.action_masks:
            batch["action_masks"] = torch.tensor(np.array(self.action_masks, dtype=np.float32).squeeze(), device=config.device)

        if batch_size == 1:
            # Add first dimension if batch has only one sample
            batch = {data_name: data[None, ...] for data_name, data in batch.items()}
        batch = TensorDict(batch, batch_size=batch_size, device=config.device)

        next_obs = torch.tensor(self.observation["observation"], dtype=torch.float32, device=config.device)
        next_done = torch.tensor(self.done, dtype=torch.float32, device=config.device)
        next_value = policy.get_value(next_obs)

        batch["advantages"] = torch.zeros_like(batch["rewards"], device=config.device)
        lastgaelam = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - batch["dones"][t + 1]
                nextvalues = batch["values"][t + 1]
            delta = batch["rewards"][t] + config.gamma * nextvalues * nextnonterminal - batch["values"][t]
            batch["advantages"][t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
        batch["returns"] = batch["advantages"] + batch["values"]

        if self.sample_masks:
            sample_mask = torch.tensor(np.array(self.sample_masks, dtype=bool), device=config.device)
            batch = batch[sample_mask]

        return batch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):      
    def __init__(self, action_space, observation_space, config):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(flatten_space(observation_space).shape[0], 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(flatten_space(observation_space).shape[0], 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, flatten_space(action_space).shape[0]), std=0.01),
        )

        self.get_value = self._get_zero_value if config.zero_value else self._get_value
        
    def _get_value(self, observations) -> Tensor:
        return self.critic(observations)

    def _get_zero_value(self, observations: Tensor) -> Tensor:
        if len(observations.shape) > 1:
            values = torch.zeros((observations.shape[0], 1), dtype=torch.float32, device=observations.device)
        else:
            values = torch.tensor([0], dtype=torch.float32, device=observations.device)
        return values

    def get_action_and_value(
        self, observations, action=None, action_masks=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits: Tensor = self.actor(observations)

        if action_masks is not None:
            inf_mask = torch.clamp(torch.log(action_masks), min=torch.finfo(action_masks.dtype).min)
            logits = logits + inf_mask

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        elif action is None and not self.training:
            action = probs.mode
        return action, probs.log_prob(action), probs.entropy(), self.get_value(observations)


class DirectDistribution(nn.Module):      
    def __init__(self, action_space, observation_space, config):
        super().__init__()
        self.config = config
        self.actor = nn.Parameter(torch.ones([flatten_space(action_space).shape[0]], dtype=torch.float32, device=config.device), requires_grad=True)
        
    def get_value(self, observations) -> Tensor:
        if len(observations.shape) > 1:
            values = torch.zeros((observations.shape[0], 1), dtype=torch.float32, device=observations.device)
        else:
            values = torch.tensor([0], dtype=torch.float32, device=observations.device)
        return values

    def get_action_and_value(
        self, observations, action=None, action_masks=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits: Tensor = F.relu(self.actor)

        if len(observations.shape) > 1:
            logits = torch.tile(logits, (observations.shape[0], 1))

        if action_masks is not None:
            inf_mask = torch.clamp(torch.log(action_masks), min=torch.finfo(action_masks.dtype).min)
            logits = logits + inf_mask

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        elif action is None and not self.training:
            action = probs.mode
        return action, probs.log_prob(action), probs.entropy(), self.get_value(observations)


class PPOPolicyModule:
    def __init__(self, agent_ids: list[str], action_space, observation_space, env, config):
        super().__init__()
        assert isinstance(agent_ids, list)
        self.agent_ids = agent_ids
        self.action_space = action_space
        self.observation_space = observation_space
        self.config = config

        self.agents = {agent_id: Agent(agent_id) for agent_id in self.agent_ids}

        self.policy = config.model_class.value(self.action_space, self.observation_space, config).to(config.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate, eps=1e-3)

    def clear_buffers(self):
        [agent.init_buffer() for agent in self.agents.values()]

    def get_actions(self, observations: dict[str, Any]):
        present_agents = [agent_id for agent_id in self.agent_ids if agent_id in observations]

        obs_list = [observations[agent_id]["observation"] for agent_id in present_agents]
        obs_array = np.array(obs_list, dtype=np.float32)
        stacked_obs = torch.tensor(obs_array, device=self.config.device)

        stacked_action_mask = None
        if "action_mask" in observations[present_agents[0]]:
            action_mask_list = [observations[agent_id]["action_mask"] for agent_id in present_agents]
            action_mask_array = np.array(action_mask_list, dtype=np.float32)
            stacked_action_mask = torch.tensor(action_mask_array, device=self.config.device)

        with torch.no_grad():
            actions, logprobs, _, values = self.policy.get_action_and_value(stacked_obs, action_masks=stacked_action_mask)
        
        for agent_id, value, action, logprob in zip(present_agents, values, actions, logprobs):
            self.agents[agent_id].add_policy_sample(action, value, logprob)

        return {agent_id: action for agent_id, action in zip(present_agents, actions.cpu().numpy())}
    
    def add_observations_dones(self, observations: dict[str, Any], dones: dict[str, Any]):
        assert observations.keys() == dones.keys()
        for agent_id, observation in observations.items():
            self.agents[agent_id].add_observation_done(observation, dones[agent_id])
    
    def add_rewards(self, rewards: dict[str, Any]):
        for agent_id, reward in rewards.items():
            self.agents[agent_id].add_reward(reward)

    def train(self):
        #NOTE: all kinds of dirty tricky here to check for empty batches, should be more rigid
        with torch.no_grad():
            batch_list = [agent.get_train_batch(self.policy, self.config) for agent in self.agents.values() if agent.has_samples]
        if not batch_list:
            print(f"WARNING: No samples gathered for agents {self.agent_ids}")
            return {}
        
        batch = torch.cat(batch_list)

        if len(batch) == 0:
            print(f"WARNING: No samples gathered for agents {self.agent_ids}")
            return {}

        # pd.DataFrame(
        #     {
        #         "actions": batch["actions"].reshape(-1).numpy(),
        #         "logprobs": batch["logprobs"].reshape(-1).numpy(),
        #         "rewards": batch["rewards"].reshape(-1).numpy(),
        #         "advantages": batch["advantages"].reshape(-1).numpy(),
        #         "returns": batch["returns"].reshape(-1).numpy(),
        #         "values": batch["values"].reshape(-1).numpy(),
        #         "dones": batch["dones"].reshape(-1).numpy(),
        #     }
        # ).to_csv(f"analysis/log_buffer_{self.agent_ids[0]}-{self.agent_ids[-1]}.csv")

        # flatten the batch
        b_obs = batch["observations"].reshape((-1,) + self.observation_space.shape)
        b_logprobs = batch["logprobs"].reshape(-1)
        b_actions = batch["actions"].reshape((-1,) + self.action_space.shape)
        if self.config.warehouse.action_masking:
            b_action_mask = batch["action_masks"].reshape((-1,) + flatten_space(self.action_space).shape)
        b_advantages = batch["advantages"].reshape(-1)
        b_returns = batch["returns"].reshape(-1)
        b_values = batch["values"].reshape(-1)

        batch_size = b_obs.shape[0]
        minibatch_size = max(batch_size // self.config.num_minibatches, 1) #NOTE: hacky way to prevent errors with little samples

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                if self.config.warehouse.action_masking:
                    _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(b_obs[mb_inds], action=b_actions.long()[mb_inds], action_masks=b_action_mask[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(b_obs[mb_inds], action=b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.config.clip_coef, self.config.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log_data = {
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "unclipped_value": newvalue.mean().item(),
            "gradient_norm": total_norm.item(),
            "total_loss": loss.item(),
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
        }

        return log_data


class MultiAgentModule:
    def __init__(self, agent_policy_mapping: dict[str, str], env: ParallelEnv, config):
        policy_agents_mapping = defaultdict(list)
        for agent_id, policy_id in agent_policy_mapping.items():
            policy_agents_mapping[policy_id].append(agent_id)

        self.policies: dict[str, PPOPolicyModule] = {}
        for policy_id, agent_ids in policy_agents_mapping.items():
            action_space = env.action_space(agent_ids[0])
            observation_space = env.observation_space(agent_ids[0])["observation"]
            self.policies[policy_id] = config.policy_module.value(agent_ids, action_space, observation_space, env, config)

    def clear_buffers(self):
        [policy.clear_buffers() for policy in self.policies.values()]

    def iter_policy_agent_dicts(self, *agent_dicts: dict[str, Any]):
        for policy in self.policies.values():
            pa_dicts = []
            for agent_dict in agent_dicts:
                pa_dict = {agent_id: agent_dict[agent_id] for agent_id in policy.agent_ids if agent_id in agent_dict}
                if not pa_dict:
                    continue
                pa_dicts.append(pa_dict)
            if not pa_dicts:
                continue
            yield policy, pa_dicts if len(pa_dicts) > 1 else pa_dicts[0]

    def get_actions(self, observations: dict[str, Any]):
        actions = {}
        for policy, policy_observations in self.iter_policy_agent_dicts(observations):
            actions.update(policy.get_actions(policy_observations))
        return actions

    def add_observations_dones(self, observations, dones):
        for policy, (policy_observations, policy_dones) in self.iter_policy_agent_dicts(observations, dones):
            policy.add_observations_dones(policy_observations, policy_dones)

    def add_rewards(self, rewards: dict[str, Any]):
        for policy, policy_rewards in self.iter_policy_agent_dicts(rewards):
            policy.add_rewards(policy_rewards)

    def train(self):
        combined_log_data = {}
        for policy_id, policy in self.policies.items():
            #TODO: parallelize?
            log_data = policy.train()
            log_data = {f"{name}/{policy_id}": data for name, data in log_data.items()}
            combined_log_data.update(log_data)

        return combined_log_data

    def save(self, directory):
        for policy_id, policy in self.policies.items():
            torch.save(policy.policy.state_dict(), f"{directory}/{policy_id}")
    
    def load(self, directory):
        for policy_id, policy in self.policies.items():
            policy.policy.load_state_dict(torch.load(f"{directory}/{policy_id}"))


class HeuristicPolicyModule:
    def __init__(self, agent_ids: list[str], action_space, observation_space, env, config):
        super().__init__()
        assert isinstance(agent_ids, list)
        self.agent_ids = agent_ids
        self.item_loc_dict = env.item_loc_dict

        if len(env.possible_agents) == 12:
            x_delivery = 24
            self.lane_assignment = {
                "AGV_1": 3,
                "AGV_2": 6,
                "AGV_3": 7,
                "AGV_4": 10,
                "AGV_5": 11,
                "AGV_6": 14,
                "AGV_7": 15,
                "AGV_8": 18,
                "PICKER_9": 4,
                "PICKER_10": 8,
                "PICKER_11": 13,
                "PICKER_12": 15,
            }
        elif len(env.possible_agents) == 21:
            x_delivery = 44
            self.lane_assignment = {
                "AGV_1": 2,
                "AGV_2": 3,
                "AGV_3": 6,
                "AGV_4": 7,
                "AGV_5": 10,
                "AGV_6": 11,
                "AGV_7": 14,
                "AGV_8": 15,
                "AGV_9": 18,
                "AGV_10": 19,
                "AGV_11": 22,
                "AGV_12": 23,
                "AGV_13": 26,
                "AGV_14": 27,
                "PICKER_15": 3,
                "PICKER_16": 7,
                "PICKER_17": 11,
                "PICKER_18": 14,
                "PICKER_19": 18,
                "PICKER_20": 22,
                "PICKER_21": 26,
            }

        x_agent_ref = x_delivery - 4
        self.delivery_assignment = {}
        for agent_id, lane_y in self.lane_assignment.items():
            if agent_id.startswith("AGV"):
                for delivery_id, location in env.item_loc_dict.items():
                    if location[1] == lane_y:
                        break
                self.delivery_assignment[agent_id] = delivery_id

        # self.lane_assignment.update({agent_id: env.item_loc_dict[delivery_id][1] for agent_id, delivery_id in self.delivery_assignment.items()})

        self.shelf_preferences = {}
        y_factor = 8
        for agent_id in agent_ids:
            shelf_preference = []
            lane_y = self.lane_assignment[agent_id]
            for item_id in range(len(env.item_loc_dict)):
                x, y = env.item_loc_dict[item_id + 1]
                if x == x_delivery:
                    continue
                shelf_preference.append(math.sqrt((x - x_agent_ref)**2 + ((y - lane_y) * y_factor)**2))
            
            shelf_preference_sorted = [sorted(shelf_preference).index(x) for x in shelf_preference]
            self.shelf_preferences[agent_id] = np.array(shelf_preference_sorted, dtype=np.int64)


    def get_actions(self, observations: dict[str, Any]):
        actions = {}
        for agent_id, observation in observations.items():
            action_mask = observation["action_mask"]

            if agent_id.startswith("AGV"):
                shelf_action_shift = 11 if len(self.lane_assignment) == 12 else 15
                to_deliver = action_mask[1]
            elif agent_id.startswith("PICKER"):
                shelf_action_shift = 1
                to_deliver = False

            shelves_mask = action_mask[shelf_action_shift:]
            any_shelves = np.any(shelves_mask)

            if any_shelves:
                action_preferences = np.where(shelves_mask, self.shelf_preferences[agent_id], 1e6)
                action = np.int64(np.argmin(action_preferences) + shelf_action_shift)
            elif to_deliver:
                action = np.int64(self.delivery_assignment[agent_id])
            else:
                action = np.int64(0)

            
            assert action in np.argwhere(action_mask).squeeze()
            actions[agent_id] = action

        return actions

