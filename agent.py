from collections import defaultdict
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import flatten_space
from pettingzoo import ParallelEnv
from tensordict import TensorDict
from torch import Tensor, nn, optim
from torch.distributions import Categorical


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.init_buffer()

    def init_buffer(self):
        self.buffer = defaultdict(list)

    def add_to_buffer(self, key, value):
        self.buffer[key].append(np.array(value).squeeze())

    def get_train_batch(self, config) -> TensorDict:
        batch_length = len(self.buffer["observations"]) - 1
        next_done = self.buffer["dones"][-1]
        next_value = self.buffer["values"][-1]

        batch = {data_name: data[:-1] for data_name, data in self.buffer.items() if data_name != "sample_mask"}
        
        batch = {data_name: torch.tensor(np.array(data, dtype=np.float32).squeeze(), device=config.device) for data_name, data in batch.items()}

        with torch.no_grad():
            advantages = torch.zeros_like(batch["rewards"], device=config.device)
            lastgaelam = 0
            for t in reversed(range(batch_length)):
                if t == batch_length - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - batch["dones"][t + 1]
                    nextvalues = batch["values"][t + 1]
                delta = batch["rewards"][t] + config.gamma * nextvalues * nextnonterminal - batch["values"][t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + batch["values"]

        batch["advantages"] = advantages
        batch["returns"] = returns

        batch = TensorDict(batch, batch_size=batch_length, device=config.device)

        if config.warehouse.sample_masking:
            sample_mask = torch.tensor(np.array(self.buffer["sample_mask"][:-1], dtype=bool), device=config.device)
            batch = batch[sample_mask]

        return batch

class PPOPolicyModule:
    def __init__(self, agent_ids: list[str], action_space, observation_space, config):
        super().__init__()
        assert isinstance(agent_ids, list)
        self.agent_ids = agent_ids
        self.action_space = action_space
        self.observation_space = observation_space
        self.config = config

        self.agents = {agent_id: Agent(agent_id) for agent_id in self.agent_ids}

        self.policy = PPO(self.action_space, self.observation_space).to(config.device)

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
            self.agents[agent_id].add_to_buffer("observations", observations[agent_id]["observation"])
            self.agents[agent_id].add_to_buffer("values", value)
            self.agents[agent_id].add_to_buffer("actions", action)
            self.agents[agent_id].add_to_buffer("logprobs", logprob)
            if "action_mask" in observations[agent_id]:
                self.agents[agent_id].add_to_buffer("action_mask", observations[agent_id]["action_mask"])
            if "sample_mask" in observations[agent_id]:
                self.agents[agent_id].add_to_buffer("sample_mask", observations[agent_id]["sample_mask"])
                

        return {agent_id: action for agent_id, action in zip(present_agents, actions.cpu().numpy())}
    
    def add_to_buffer(self, data_dict: dict[str, Any]):
        for data_name, agent_dict in data_dict.items():
            for agent_id, data in agent_dict.items():
                self.agents[agent_id].add_to_buffer(data_name, data)

    def train(self):
        # bootstrap value if not done
        batch_list = [agent.get_train_batch(self.config) for agent in self.agents.values()]
        batch = torch.cat(batch_list)

        # flatten the batch
        b_obs = batch["observations"].reshape((-1,) + self.observation_space.shape)
        b_logprobs = batch["logprobs"].reshape(-1)
        b_actions = batch["actions"].reshape((-1,) + self.action_space.shape)
        if self.config.warehouse.action_masking:
            b_action_mask = batch["action_mask"].reshape((-1,) + flatten_space(self.action_space).shape)
        b_advantages = batch["advantages"].reshape(-1)
        b_returns = batch["returns"].reshape(-1)
        b_values = batch["values"].reshape(-1)

        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // self.config.num_minibatches

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                if self.config.warehouse.action_masking:
                    _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(b_obs[mb_inds], action=b_actions.long()[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(b_obs[mb_inds], action=b_actions.long()[mb_inds], action_masks=b_action_mask[mb_inds])
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
            "measures/batch_size": batch_size,
            "measures/minibatch_size": minibatch_size,
            "measures/unclipped_value": newvalue.mean().item(),
            "measures/gradient_norm": total_norm.item(),
            "losses/total_loss": loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "measures/old_approx_kl": old_approx_kl.item(),
            "measures/approx_kl": approx_kl.item(),
            "measures/clipfrac": np.mean(clipfracs),
            "measures/explained_variance": explained_var,
        }

        return log_data


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):      
    def __init__(self, action_space, observation_space):
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

    def get_value(self, observations) -> Tensor:
        return self.critic(observations)

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
        return action, probs.log_prob(action), probs.entropy(), self.critic(observations)


class MultiAgentModule:
    def __init__(self, agent_policy_mapping: dict[str, str], env: ParallelEnv, config):
        self.agent_policy_mapping = agent_policy_mapping
        
        policy_agents_mapping = defaultdict(list)
        for agent_id, policy_id in agent_policy_mapping.items():
            policy_agents_mapping[policy_id].append(agent_id)

        self.policies: dict[str, PPOPolicyModule] = {}
        for policy_id, agent_ids in policy_agents_mapping.items():
            action_space = env.action_space(agent_ids[0])
            observation_space = env.observation_space(agent_ids[0])["observation"]
            self.policies[policy_id] = PPOPolicyModule(agent_ids, action_space, observation_space, config)

    def clear_buffers(self):
        [policy.clear_buffers() for policy in self.policies.values()]

    def iter_policy_agent_dict(self, agent_dict: dict[str, Any]):
        for policy_id, policy in self.policies.items():
            policy_agent_dict = {agent_id: data for agent_id, data in agent_dict.items() if self.agent_policy_mapping[agent_id] == policy_id}
            yield policy, policy_agent_dict

    def get_actions(self, observations: dict[str, Any]):
        actions = {}
        for policy, policy_observations in self.iter_policy_agent_dict(observations):
            actions.update(policy.get_actions(policy_observations))

        return actions

    def add_to_buffer(self, data_dict):
        for data_name, agent_dict in data_dict.items():
            for policy, policy_agent_dict in self.iter_policy_agent_dict(agent_dict):
                policy.add_to_buffer({data_name: policy_agent_dict})

    def train(self):
        combined_log_data = {}
        for policy_id, policy_module in self.policies.items():
            #TODO: parallelize?
            log_data = policy_module.train()
            log_data = {f"{name}/{policy_id}": data for name, data in log_data.items()}
            combined_log_data.update(log_data)

        return combined_log_data

    def save(self, directory):
        for policy_id, policy in self.policies.items():
            torch.save(policy.policy.state_dict(), f"{directory}/{policy_id}")