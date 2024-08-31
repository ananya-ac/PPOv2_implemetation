import torch.nn as nn
import torch



def layer_initilization(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class PPOAgent(nn.Module):
    
    def __init__(self, n_actions, n_frames = 4 ):
        super(PPOAgent, self).__init__()
        
        self.conv = nn.Sequential(
            layer_initilization(nn.Conv2d(n_frames, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_initilization(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_initilization(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512)
        )
        
        self.actor = nn.Sequential(
            layer_initilization(nn.Linear(512, n_actions))
        )
        
        self.critic = nn.Sequential(
            layer_initilization(nn.Linear(512, 1))
        )
        
        
        def get_value(self, x):
            return self.critic(self.conv(x))
        
        def get_action(self, x, action = None):
            logits = self.actor(self.conv(x))
            dist = torch.distributions.Categorical(logits=logits)
            if action == None:
                action = dist.sample()
            value = self.critic(self.conv(x))
            return action, dist.log_prob(), dist.entropy(), value
        
        

def update_agent(agent, optimizer, observations, actions, returns, values, advantages, logprobs, dones, truncs, clip_param = 0.2, vf_coef = 0.5, ent_coef = 0.01):
    
    observations = observations.view(-1, *observations.shape[2:])
    actions = actions.view(-1, *actions.shape[2:])
    returns = returns.view(-1)
    values = values.view(-1)
    advantages = advantages.view(-1)
    logprobs = logprobs.view(-1)
    
    for _ in range(num_epochs):
        for idx in range(0, observations.size(0), batch_size):
            batch_indices = slice(idx, idx + batch_size)
            obs_batch = observations[batch_indices]
            act_batch = actions[batch_indices]
            ret_batch = returns[batch_indices]
            adv_batch = advantages[batch_indices]
            norm_adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
            logprob_batch = logprobs[batch_indices]
            
            _, logprob, entropy, value = agent.get_action(obs_batch, act_batch)
            ratio = (logprob - logprob_batch).exp()
            policy_loss = -torch.min(ratio * norm_adv_batch, torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * norm_adv_batch).mean()
            value_loss = 0.5 * (value - ret_batch).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_loss * vf_coef - entropy_loss * ent_coef
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()

            