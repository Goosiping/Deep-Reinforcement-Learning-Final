import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Normal
from enum import Enum
import time

class TYPE(Enum):
    discrete = 0
    continuous = 1
    multibinary = 2

class Critic2Heads(nn.Module):
    def __init__(self, input_dim):
        super(Critic2Heads, self).__init__()
        self.ext = nn.Linear(input_dim, 1)
        self.int = nn.Linear(input_dim, 1)

        init_orthogonal(self.ext, 0.01)
        init_orthogonal(self.int, 0.01)

    def forward(self, x):
        ext_value = self.ext(x)
        int_value = self.int(x)
        return torch.cat([ext_value, int_value], dim=1).squeeze(-1)

    @property
    def weight(self):
        return self.ext.weight, self.int.weight

    @property
    def bias(self):
        return self.ext.bias, self.int.bias

def one_hot_code(values, value_dim):
    code = torch.zeros((values.shape[0], value_dim), dtype=torch.float32, device=values.device)
    code = code.scatter(1, values, 1.0)
    return code

class DiscreteHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DiscreteHead, self).__init__()
        self.logits = nn.Linear(input_dim, action_dim, bias=True)

        torch.nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, x):
        logits = self.logits(x)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)

        action = dist.sample().unsqueeze(1)

        return action, probs

    @staticmethod
    def log_prob(probs, actions):
        actions = torch.argmax(actions, dim=1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions).unsqueeze(1)

        return log_prob

    @staticmethod
    def entropy(probs):
        dist = Categorical(probs)
        entropy = -dist.entropy()
        return entropy.mean()

    @property
    def weight(self):
        return self.logits.weight

    @property
    def bias(self):
        return self.logits.bias

class Actor(nn.Module):
    def __init__(self, model, head, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.head_type = head
        self.head = None
        self.head = DiscreteHead


        self.model = model

    def forward(self, x):
        return self.model(x)

    def log_prob(self, probs, actions):
        return self.head.log_prob(probs, actions)

    def entropy(self, probs):
        return self.head.entropy(probs)

    def encode_action(self, action):
        return one_hot_code(action, self.action_dim)

def __init_general(function, layer, gain):
    if type(layer.weight) is tuple:
        for w in layer.weight:
            function(w, gain)
    else:
        function(layer.weight, gain)

    if type(layer.bias) is tuple:
        for b in layer.bias:
            nn.init.zeros_(b)
    else:
        nn.init.zeros_(layer.bias)

def init_orthogonal(layer, gain=1.0):
    __init_general(nn.init.orthogonal_, layer, gain)

class PPOAtariNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, head):
        super(PPOAtariNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        input_channels = self.input_shape[0]
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512

        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim)
        )

        init_orthogonal(self.features[0], np.sqrt(2))
        init_orthogonal(self.features[2], np.sqrt(2))
        init_orthogonal(self.features[4], np.sqrt(2))
        init_orthogonal(self.features[6], np.sqrt(2))
        init_orthogonal(self.features[9], np.sqrt(2))

        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1)
        )

        init_orthogonal(self.critic[1], 0.1)
        init_orthogonal(self.critic[3], 0.01)

        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            DiscreteHead(self.feature_dim, action_dim)
        )

        init_orthogonal(self.actor[1], 0.01)
        init_orthogonal(self.actor[3], 0.01)

        self.actor = Actor(self.actor, head, self.action_dim)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs
    
class RunningStatsSimple:
    def __init__(self, shape, device):
        self.count = 1
        self.eps = 0.0000001
        self.mean = torch.zeros(shape, device=device)
        self.var = 0.01 * torch.ones(shape, device=device)
        self.std = (self.var ** 0.5) + self.eps

    def update(self, x):
        self.count += 1

        mean = self.mean + (x.mean(axis=0) - self.mean) / self.count
        var = self.var + ((x - self.mean) * (x - mean)).mean(axis=0)

        self.mean = mean
        self.var = var

        self.std = ((self.var / self.count) ** 0.5) + self.eps
    
class RNDModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, device):
        super(RNDModelAtari, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        input_channels = 1
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512
        self.state_average = RunningStatsSimple((4, input_height, input_width), device)

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.target_model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim)
        )

        init_orthogonal(self.target_model[0], np.sqrt(2))
        init_orthogonal(self.target_model[2], np.sqrt(2))
        init_orthogonal(self.target_model[4], np.sqrt(2))
        init_orthogonal(self.target_model[7], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        init_orthogonal(self.model[0], np.sqrt(2))
        init_orthogonal(self.model[2], np.sqrt(2))
        init_orthogonal(self.model[4], np.sqrt(2))
        init_orthogonal(self.model[7], np.sqrt(2))
        init_orthogonal(self.model[9], np.sqrt(2))
        init_orthogonal(self.model[11], np.sqrt(2))

    def prepare_input(self, state):
        x = state - self.state_average.mean
        return x[:, 0, :, :].unsqueeze(1)

    def forward(self, state):
        x = self.prepare_input(state)
        predicted_code = self.model(x)
        target_code = self.target_model(x)
        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)
            error = torch.sum(torch.pow(target - prediction, 2), dim=1).unsqueeze(-1) / 2

        return error

    def loss_function(self, state):
        prediction, target = self(state)
        # loss = nn.functional.mse_loss(self(state), self.encode(state).detach(), reduction='none').sum(dim=1)
        loss = torch.pow(target - prediction, 2)
        mask = torch.rand_like(loss) < 0.25
        loss *= mask
        loss = loss.sum() / mask.sum()

        # analytic = ResultCollector()
        # analytic.update(loss_prediction=loss.unsqueeze(-1).detach())

        return loss

    def update_state_average(self, state):
        self.state_average.update(state)

class PPOAtariMotivationNetwork(PPOAtariNetwork):
    def __init__(self, input_shape, action_dim, device):
        super(PPOAtariMotivationNetwork, self).__init__(input_shape, action_dim, device)

        self.critic = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            Critic2Heads(self.feature_dim)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)

class PPOAtariNetworkRND(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, device):
        super(PPOAtariNetworkRND, self).__init__(input_shape, action_dim, device)
        self.rnd_model = RNDModelAtari(input_shape, self.action_dim, device)

class RNDMotivation:
    def __init__(self, network, lr, eta=1, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
        self._device = device
        self.reward_stats = RunningStats(1, device)

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("RND motivation training time {0:.2f}s".format(end - start))


    def error(self, state0):
        return self._network.error(state0)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)

        states = sample.state.to(self._device)

        return self.reward(states)

    def reward(self, state0):
        reward = self.error(state0)
        return reward * self._eta

    def update_state_average(self, state):
        self._network.update_state_average(state)

    def update_reward_average(self, reward):
        self.reward_stats.update(reward.to(self._device))

class RunningStats:
    def __init__(self, shape, device, n=1):
        self.n = n
        if n > 1:
            shape = (n,) + shape
            self.count = torch.ones((n, 1), device=device)
        else:
            self.count = 1
        self.eps = 0.0000001
        self.max = torch.zeros(shape, device=device)
        self.sum = torch.zeros(shape, device=device)
        self.mean = torch.zeros(shape, device=device)
        self.var = 0.01 * torch.ones(shape, device=device)
        self.std = (self.var ** 0.5) + self.eps

    def update(self, x, reduction='mean'):
        self.count += 1

        mean = None
        var = None
        max = torch.maximum(self.max, x)

        if reduction == 'mean':
            self.sum += x.mean(axis=0)
            mean = self.mean + (x.mean(axis=0) - self.mean) / self.count
            var = self.var + ((x - self.mean) * (x - mean)).mean(axis=0)
        if reduction == 'none':
            self.sum += x
            mean = self.mean + (x - self.mean) / self.count
            var = self.var + ((x - self.mean) * (x - mean))

        self.max = max
        self.mean = mean
        self.var = var

        self.std = ((self.var / self.count) ** 0.5) + self.eps

    def reset(self, i):
        if self.n > 1:
            self.max[i].fill_(0)
            self.sum[i].fill_(0)
            self.mean[i].fill_(0)
            self.var[i].fill_(0.01)
            self.count[i] = 1
        else:
            self.max.fill_(0)
            self.sum.fill_(0)
            self.mean.fill_(0)
            self.var.fill_(0.01)
            self.count = 1