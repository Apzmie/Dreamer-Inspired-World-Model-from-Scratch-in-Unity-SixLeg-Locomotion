from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter       

BASE_DIR = ""


class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        
    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                                    
                    state = self.state[p]                   
                    if len(state) == 0:
                        state["step"] = 0
                        state["grad_momentum"] = torch.zeros_like(p)
                        state["grad_squared_momentum"] = torch.zeros_like(p)
                        
                    grad_momentum = state["grad_momentum"]
                    grad_squared_momentum = state["grad_squared_momentum"]
                    
                    state["step"] += 1
                    step = state["step"]
                    
                    grad = p.grad
                    
                    grad_momentum.mul_(beta1)
                    grad_momentum.add_(grad, alpha=1.0 - beta1)
                    
                    grad_squared_momentum.mul_(beta2)
                    grad_squared_momentum.addcmul_(grad, grad, value=1.0 - beta2)
                    
                    bias_correction1 = grad_momentum / (1.0 - beta1 ** step)
                    bias_correction2 = grad_squared_momentum / (1.0 - beta2 ** step)
        
                    p.addcdiv_(bias_correction1, torch.sqrt(bias_correction2) + eps, value=-lr)   
                    
                    
class LaProp(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.99), eps=1e-20):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        
    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                        
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["grad_momentum"] = torch.zeros_like(p)
                        state["grad_squared_momentum"] = torch.zeros_like(p)
                        
                    grad_momentum = state["grad_momentum"]
                    grad_squared_momentum = state["grad_squared_momentum"]
                    
                    state["step"] += 1
                    step = state["step"]
                    
                    grad = p.grad
                    
                    grad_squared_momentum.mul_(beta2)
                    grad_squared_momentum.addcmul_(grad, grad, value=1.0 - beta2)
                    bias_correction2 = grad_squared_momentum / (1.0 - beta2 ** step)
                    
                    normalized_grad = grad / (torch.sqrt(bias_correction2) + eps)
                    
                    grad_momentum.mul_(beta1)
                    grad_momentum.add_(normalized_grad, alpha=1.0 - beta1)
                    bias_correction1 = grad_momentum / (1.0 - beta1 ** step)
                    
                    p.add_(bias_correction1, alpha=-lr)  
                    
                    
class NextStateModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.next_state = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        next_state = self.next_state(x)       
        return next_state  
        
        
class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        
        nn.init.zeros_(self.reward.weight)
        nn.init.zeros_(self.reward.bias)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        reward = self.reward(x)
        return reward
        

class ContinueModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.continue_logit = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        logit = self.continue_logit(x)
        return logit
        

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.next_state_model = NextStateModel(state_dim, action_dim)        
        self.reward_model = RewardModel(state_dim, action_dim)
        self.continue_model = ContinueModel(state_dim, action_dim)    
    
    def loss(self, state, action, next_state, reward, continuation):
        pred_next_state = self.next_state_model(state, action)
        pred_reward = self.reward_model(state, action)
        pred_continue_logit = self.continue_model(state, action)
        
        next_state_loss = F.mse_loss(pred_next_state, next_state)                
        reward_loss = F.mse_loss(pred_reward, reward)        
        continue_loss = F.binary_cross_entropy_with_logits(pred_continue_logit, continuation)
        
        total_loss = next_state_loss + reward_loss + continue_loss
        
        return total_loss, next_state_loss, reward_loss, continue_loss
        
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.silu(self.fc1(state))
        x = F.silu(self.fc2(x))       
        mean = self.mean(x)       
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()       
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
        
    def deterministic(self, state):
        mean, _ = self.forward(state)
        action = torch.tanh(mean)
        return action
        

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        
    def forward(self, state):
        x = F.silu(self.fc1(state))
        x = F.silu(self.fc2(x))
        value = self.value(x)
        return value
        
        
class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.world_model = WorldModel(state_dim, action_dim)
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.target_critic = Critic(state_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.world_model_optimizer = LaProp(self.world_model.parameters(), lr=lr)        
        self.critic_optimizer = LaProp(self.critic.parameters(), lr=lr)   
        
        ###########################################
        ### Load Actor (fc1, fc2, mean) ###
        ###########################################
        
        #state_dict = torch.load(f"{BASE_DIR}/previous_model.pth")
        #self.actor.fc1.load_state_dict({"weight": state_dict["fc1.weight"], "bias": state_dict["fc1.bias"]})
        #self.actor.fc2.load_state_dict({"weight": state_dict["fc2.weight"], "bias": state_dict["fc2.bias"]})
        #self.actor.mean.load_state_dict({"weight": state_dict["mean.weight"], "bias": state_dict["mean.bias"]})
        
        #with torch.no_grad():        
        #    self.actor.log_std.weight.zero_()
        #    self.actor.log_std.bias.fill_(-2)        
        
        #==========================================
        
        self.actor_optimizer = LaProp(self.actor.parameters(), lr=lr)
        
    def imagine(self, state, horizon=2):
        imagined_states = []
        imagined_actions = []
        imagined_rewards = []
        imagined_next_states = []
        imagined_continuations = []
        imagined_log_probs = []
        
        for t in range(horizon):
            action, log_prob = self.actor.sample(state)
            
            next_state = self.world_model.next_state_model(state, action) 
            reward = self.world_model.reward_model(state, action)
            continue_logit = self.world_model.continue_model(state, action)
            continuation = torch.sigmoid(continue_logit) 
            
            imagined_states.append(state)
            imagined_actions.append(action)
            imagined_rewards.append(reward)
            imagined_next_states.append(next_state)
            imagined_continuations.append(continuation)
            imagined_log_probs.append(log_prob)
            
            state = next_state
        
        imagined_states = torch.stack(imagined_states, dim=1)    
        imagined_actions = torch.stack(imagined_actions, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_next_states = torch.stack(imagined_next_states, dim=1)
        imagined_continuations = torch.stack(imagined_continuations, dim=1)
        imagined_log_probs = torch.stack(imagined_log_probs, dim=1)
       
        return imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_continuations, imagined_log_probs
        
    def compute_return(self, rewards, next_values, continuations, gamma=0.997, lambda_=0.95):
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_returns = next_values[:, -1, :]
        
        for t in reversed(range(H)):    
            next_returns = rewards[:, t, :] + gamma * continuations[:, t, :] * ((1 - lambda_) * next_values[:, t, :] + lambda_ * next_returns)            
            returns[:, t, :] = next_returns
            
        return returns
        
    def target_critic_update(self, tau=0.005):
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.mul_(1.0 - tau)
                target_param.add_(tau * param.data)  
                
    def adaptive_grad_clip(self, parameters, clip_factor=0.3, eps=1e-3):
        with torch.no_grad():
            for p in parameters:
                if p.grad is None:
                    continue

                param_norm = torch.norm(p)
                grad_norm = torch.norm(p.grad)

                max_norm = (param_norm + eps) * clip_factor
                if grad_norm > max_norm:
                    p.grad.mul_(max_norm / (grad_norm + 1e-6))
                
    def world_model_update(self, batch, num_updates=8):
        state = torch.from_numpy(batch['state'])
        action = torch.from_numpy(batch['action'])
        reward = torch.from_numpy(batch['reward'])
        next_state = torch.from_numpy(batch['next_state'])
        done = torch.from_numpy(batch['done'])
        continuation = 1.0 - done
        
        for _ in range(num_updates):
            world_model_loss, next_state_loss, reward_loss, continue_loss = self.world_model.loss(state, action, next_state, reward, continuation)
            self.world_model_optimizer.zero_grad()
            world_model_loss.backward()
            self.adaptive_grad_clip(self.world_model.parameters())
            self.world_model_optimizer.step()       
        
        return {
            "next_state_loss": next_state_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
        }

    def update(self, batch, eta=1e-4):
        state = torch.from_numpy(batch['state'])
        action = torch.from_numpy(batch['action'])
        reward = torch.from_numpy(batch['reward'])
        next_state = torch.from_numpy(batch['next_state'])
        done = torch.from_numpy(batch['done'])
        
        #==========================================
        
        for p in self.world_model.parameters():
            p.requires_grad = False 
        for p in self.actor.parameters():
            p.requires_grad = False 
            
        imagined_states, _, imagined_rewards, imagined_next_states, imagined_continuations, _ = self.imagine(state)
        imagined_values = self.critic(imagined_states)
        with torch.no_grad():
            imagined_next_values = self.target_critic(imagined_next_states)
            returns = self.compute_return(imagined_rewards, imagined_next_values, imagined_continuations)
            
        critic_loss = F.mse_loss(imagined_values, returns)        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.adaptive_grad_clip(self.critic.parameters())
        self.critic_optimizer.step()
        
        self.target_critic_update()
        
        for p in self.world_model.parameters():
            p.requires_grad = True 
        for p in self.actor.parameters():
            p.requires_grad = True 
        
        #==========================================
        
        for p in self.world_model.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False  
            
        _, _, imagined_rewards, imagined_next_states, imagined_continuations, imagined_log_probs = self.imagine(state)
        imagined_next_values = self.critic(imagined_next_states)
        returns = self.compute_return(imagined_rewards, imagined_next_values, imagined_continuations)
        
        actor_loss = -returns.mean() + eta * imagined_log_probs.mean()        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.adaptive_grad_clip(self.actor.parameters())
        self.actor_optimizer.step()
        
        for p in self.world_model.parameters():
            p.requires_grad = True
        for p in self.critic.parameters():
            p.requires_grad = True  
            
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }
        
    def save_checkpoint(self, path, buffer, online_buffer):
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            
            "buffer": buffer,
            "online_buffer": online_buffer,
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])

        self.world_model_optimizer.load_state_dict(checkpoint["world_model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        buffer = checkpoint["buffer"]
        online_buffer = checkpoint["online_buffer"]

        return buffer, online_buffer


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), batch_size=256):
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)

        return {
            "state": self.state[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_state": self.next_state[idx],
            "done": self.done[idx],
        }
    

if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    channel2 = EngineConfigurationChannel()
    channel2.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    test_env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel2], no_graphics=True, worker_id=1)
    env.reset()
    test_env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    t_behavior_name = list(test_env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    agent = Agent(state_dim, action_dim)
    buffer = ReplayBuffer(state_dim, action_dim, batch_size=192)
    online_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e3), batch_size=64)
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    #buffer, online_buffer = agent.load_checkpoint(f"{BASE_DIR}/checkpoint.pth")
    
    random_exploration_steps = 1000
    learning_starts = 500
    ac_update_start = 5000
    ac_update_interval = 4
    test_interval = 1000
    test_max_step = 1000
    
    total_steps = 0
    update_count = 0
    save_idx = 0
    best_test_reward = -float('inf')
    world_model_steps = 0
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)  
            
            if total_steps < random_exploration_steps:
                actions = np.random.uniform(low=-1.0, high=1.0, size=(len(agent_ids), action_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    actions, _ = agent.actor.sample(states_tensor)   
                actions = actions.cpu().numpy().astype(np.float32)
                
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
            
        env.step()
        next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            state = states_tensor[i].cpu().numpy()
            action = actions[i]

            if agent_id in next_terminal_steps:
                reward = next_terminal_steps[agent_id].reward
                done = 1.0
                next_state = next_terminal_steps[agent_id].obs[0]
            elif agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                done = 0.0
                next_state = next_decision_steps[agent_id].obs[0]
            else:
                continue
                
            buffer.add(state, action, reward, next_state, done)
            online_buffer.add(state, action, reward, next_state, done)
            total_steps += 1
        
        if total_steps >= learning_starts:
            batch = buffer.sample()
            online_batch = online_buffer.sample()
            batch = {
                "state": np.concatenate([batch["state"], online_batch["state"]], axis=0),
                "action": np.concatenate([batch["action"], online_batch["action"]], axis=0),
                "reward": np.concatenate([batch["reward"], online_batch["reward"]], axis=0),
                "next_state": np.concatenate([batch["next_state"], online_batch["next_state"]], axis=0),
                "done": np.concatenate([batch["done"], online_batch["done"]], axis=0),
            }
            wm_metrics = agent.world_model_update(batch)
            world_model_steps += 1

            if world_model_steps >= ac_update_start and world_model_steps % ac_update_interval == 0:
                batch = buffer.sample()
                online_batch = online_buffer.sample() 
                batch = {
                    "state": np.concatenate([batch["state"], online_batch["state"]], axis=0),
                    "action": np.concatenate([batch["action"], online_batch["action"]], axis=0),
                    "reward": np.concatenate([batch["reward"], online_batch["reward"]], axis=0),
                    "next_state": np.concatenate([batch["next_state"], online_batch["next_state"]], axis=0),
                    "done": np.concatenate([batch["done"], online_batch["done"]], axis=0),
                }
                metrics = agent.update(batch) 
                update_count += 1
                for k, v in wm_metrics.items():
                    writer.add_scalar(f"Train/{k}", v, update_count)            
                for k, v in metrics.items():
                    writer.add_scalar(f"Train/{k}", v, update_count)               
             
                if update_count % test_interval == 0:
                    print(f"Update Count {update_count}")
                    test_env.reset()
                    t_decision_steps, _ = test_env.get_steps(t_behavior_name)
                    n_test_agents = len(t_decision_steps.agent_id)
                    test_rewards = np.zeros(n_test_agents)
                    test_episode_dones = np.zeros(n_test_agents, dtype=bool)
                    test_id_to_index = {agent_id: i for i, agent_id in enumerate(t_decision_steps.agent_id)}
                 
                    test_max_step_count = 0
                    while not np.all(test_episode_dones) and test_max_step_count < test_max_step:
                        t_agent_ids = t_decision_steps.agent_id
                     
                        if len(t_agent_ids) > 0:
                            t_states_tensor = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32)                        
                            with torch.no_grad():
                                t_actions = agent.actor.deterministic(t_states_tensor)                    
                            t_actions = t_actions.cpu().numpy().astype(np.float32)
                         
                            for j, agent_id in enumerate(t_agent_ids):
                                idx = test_id_to_index[agent_id]
                                if test_episode_dones[idx]:
                                    t_actions[j] = np.zeros(action_dim)
                                
                            test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions))
                         
                        test_env.step()
                        test_max_step_count += 1
                        t_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name)
                     
                        for j, agent_id in enumerate(t_terminal_steps.agent_id):
                            i = test_id_to_index[agent_id]
                            if not test_episode_dones[i]:
                                test_rewards[i] += t_terminal_steps.reward[j]
                                test_episode_dones[i] = True

                        for j, agent_id in enumerate(t_decision_steps.agent_id):
                            i = test_id_to_index[agent_id]
                            if not test_episode_dones[i]:
                                test_rewards[i] += t_decision_steps.reward[j]
                             
                    test_average_reward = np.mean(test_rewards)
                    writer.add_scalar("Test/Average_Reward", test_average_reward, update_count)
                    print(f"{test_average_reward:.4f}")
                    torch.save(agent.actor.state_dict(), f"{BASE_DIR}/period_model.pth")
                    agent.save_checkpoint(f"{BASE_DIR}/checkpoint.pth", buffer, online_buffer)
                         
                    if test_average_reward > best_test_reward:
                        best_test_reward = test_average_reward
                        save_idx += 1
                        torch.save(agent.actor.state_dict(), f"{BASE_DIR}/#({save_idx})best_{best_test_reward:.4f}.pth") 
                        print(f"[Test] Model saved at new best reward {best_test_reward:.4f}")
                        
