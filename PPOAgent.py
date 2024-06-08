import torch
from Memory import ReplayBuffer
from Networks import PPOAtariNetwork, PPOAtariNetworkRND, RNDMotivation
import time

class Agent:
    def __init__(self, 
                 state_dim,
                 action_dim,
                 trajectory_size,
                 batch_size,
                 n_env,
                 device,
                 lr,
                 actor_loss_weight,
                 critic_loss_weight,
                 beta,
                 gamma,
                 ppo_epochs,
                 motivation_lr,
                 motivation_eta,
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.network = PPOAtariNetworkRND(state_dim, action_dim, device).to(device)
        self.motivation = RNDMotivation(self.network.rnd_model, motivation_lr, motivation_eta, device)
        self.memory = ReplayBuffer(trajectory_size, batch_size, n_env)
        self.algorithm = PPO(self.network, 
                             lr,
                             actor_loss_weight,
                             critic_loss_weight,
                             batch_size,
                             trajectory_size,
                             beta,
                             gamma, ext_adv_scale=2, int_adv_scale=1, 
                             ppo_epochs=ppo_epochs, n_env=n_env,
                             device=device, motivation=True)
        
    def get_action(self, state):
        value, action, probs = self.network(state)
        return value.detach(), action, probs.detach()
    
    def convert_action(self, action):
        return torch.argmax(action, dim=1).numpy()
    
    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), probs=probs0.cpu(), next_state=state1.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()

    def process_state(self, state):
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location='cpu'))

class PPO:
    def __init__(self, network, lr, actor_loss_weight, critic_loss_weight, batch_size, trajectory_size, p_beta, p_gamma,
                 ppo_epochs=10, p_epsilon=0.1, p_lambda=0.95, ext_adv_scale=1, int_adv_scale=1, device='cpu', n_env=1, motivation=False):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._beta = p_beta
        self._gamma = [float(g) for g in p_gamma.split(',')]
        self._epsilon = p_epsilon
        self._lambda = p_lambda
        self._device = device
        self._batch_size = batch_size
        self._trajectory_size = trajectory_size
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight

        self._trajectory = []
        self._ppo_epochs = ppo_epochs
        self._motivation = motivation
        self._n_env = n_env

        self.ext_adv_scale = ext_adv_scale
        self.int_adv_scale = int_adv_scale

        if self._n_env > 1:
            self._trajectories = [[] for _ in range(self._n_env)]

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample = memory.sample(indices, False)

            states = sample.state
            values = sample.value
            actions = sample.action
            probs = sample.probs
            rewards = sample.reward
            dones = sample.mask

            if self._motivation:
                ext_reward = rewards[:, :, 0].unsqueeze(-1)
                int_reward = rewards[:, :, 1].unsqueeze(-1)

                ext_ref_values, ext_adv_values = self.calc_advantage(values[:, :, 0].unsqueeze(-1), ext_reward, dones, self._gamma[0], self._n_env)
                int_ref_values, int_adv_values = self.calc_advantage(values[:, :, 1].unsqueeze(-1), int_reward, dones, self._gamma[1], self._n_env)
                ref_values = torch.cat([ext_ref_values, int_ref_values], dim=2)

                adv_values = ext_adv_values * self.ext_adv_scale + int_adv_values * self.int_adv_scale

            else:
                ref_values, adv_values = self.calc_advantage(values, rewards, dones, self._gamma[0], self._n_env)
                adv_values *= self.ext_adv_scale

            permutation = torch.randperm(self._trajectory_size)

            states = states.reshape(-1, *states.shape[2:])[permutation]
            actions = actions.reshape(-1, *actions.shape[2:])[permutation]
            probs = probs.reshape(-1, *probs.shape[2:])[permutation]
            adv_values = adv_values.reshape(-1, *adv_values.shape[2:])[permutation]
            ref_values = ref_values.reshape(-1, *ref_values.shape[2:])[permutation]

            self._train(states, actions, probs, adv_values, ref_values)

            end = time.time()
            print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self._trajectory_size, self._batch_size, self._ppo_epochs, end - start))

    def _train(self, states, actions, probs, adv_values, ref_values):
        # adv_values = (adv_values - torch.mean(adv_values)) / (torch.std(adv_values) + 1e-8)

        for epoch in range(self._ppo_epochs):
            for batch_ofs in range(0, self._trajectory_size, self._batch_size):
                batch_l = batch_ofs + self._batch_size
                states_v = states[batch_ofs:batch_l].to(self._device)
                actions_v = actions[batch_ofs:batch_l].to(self._device)
                probs_v = probs[batch_ofs:batch_l].to(self._device)
                batch_adv_v = adv_values[batch_ofs:batch_l].to(self._device)
                batch_ref_v = ref_values[batch_ofs:batch_l].to(self._device)

                self._optimizer.zero_grad()
                loss = self.calc_loss(states_v, batch_ref_v, batch_adv_v, actions_v, probs_v)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=0.5)
                self._optimizer.step()

    def calc_loss(self, states, ref_value, adv_value, old_actions, old_probs):
        values, _, probs = self._network(states)

        if self._motivation:
            ext_value = values[:, 0]
            int_value = values[:, 1]
            ext_ref_value = ref_value[:, 0]
            int_ref_value = ref_value[:, 1]

            loss_ext_value = torch.nn.functional.mse_loss(ext_value, ext_ref_value)
            loss_int_value = torch.nn.functional.mse_loss(int_value, int_ref_value)
            loss_value = loss_ext_value + loss_int_value
        else:
            loss_value = torch.nn.functional.mse_loss(values, ref_value)

        log_probs = self._network.actor.log_prob(probs, old_actions)
        old_logprobs = self._network.actor.log_prob(old_probs, old_actions)

        ratio = torch.exp(log_probs - old_logprobs)
        p1 = ratio * adv_value
        p2 = torch.clamp(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * adv_value
        loss_policy = -torch.min(p1, p2)
        loss_policy = loss_policy.mean()

        entropy = self._network.actor.entropy(probs)
        loss = loss_value * self._critic_loss_weight + loss_policy * self._actor_loss_weight + self._beta * entropy

        return loss

    def calc_advantage(self, values, rewards, dones, gamma, n_env):
        buffer_size = rewards.shape[0]

        returns = torch.zeros((buffer_size, n_env, 1))
        advantages = torch.zeros((buffer_size, n_env, 1))

        last_gae = torch.zeros(n_env, 1)

        for n in reversed(range(buffer_size - 1)):
            delta = rewards[n] + dones[n] * gamma * values[n + 1] - values[n]
            last_gae = delta + dones[n] * gamma * self._lambda * last_gae

            returns[n] = last_gae + values[n]
            advantages[n] = last_gae

        return returns, advantages
