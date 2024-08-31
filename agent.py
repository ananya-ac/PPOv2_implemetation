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
            