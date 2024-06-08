import threading
import multiprocessing
import time
from concurrent.futures.thread import ThreadPoolExecutor
from collections import namedtuple
import numpy
import gym
import torch
from PIL import Image
import numpy as np
import itertools

MDP_Transition = namedtuple('MDP_Transition', ('state', 'action', 'next_state', 'reward', 'mask'))

class RunningAverage:
    def __init__(self):
        self._cma = 0
        self._n = 0

    def update(self, value):
        self._cma = self._cma + (value - self._cma) / (self._n + 1)
        self._n += 1

    def value(self):
        return self._cma

class PPOTimeEstimator:
    def __init__(self, steps):
        self.total_steps = steps
        self.remaining_steps = steps
        self.remaining_time = None

        self.time_per_step = RunningAverage()
        self.start = time.time()
        self.end = time.time()

    def update(self, step=1):
        self.end = time.time()
        self.remaining_steps -= step
        self.time_per_step.update((self.end - self.start) / step)
        self.remaining_time = self.time_per_step.value() * self.remaining_steps
        self.start = time.time()

    def __str__(self):
        remaining_time = self.remaining_time
        hours = int(remaining_time // 3600)
        remaining_time -= hours * 3600
        minutes = int(remaining_time // 60)
        remaining_time -= minutes * 60
        seconds = int(remaining_time)

        return 'Progress: {0:.0f}% ETA: {1:d}:{2:02d}:{3:02d}'.format((1 - self.remaining_steps / self.total_steps) * 100, hours, minutes, seconds)

class RunningAverageWindow:
    def __init__(self, window=1, size=1):
        self._cma = np.zeros((window, size))
        self._n = 0
        self._window = window

    def update(self, value):
        self._cma[self._n] = value
        self._n += 1
        if self._n == self._window:
            self._n = 0

    def value(self):
        return self._cma.mean(axis=0)

def WrapperHardAtari(env, height=96, width=96, frame_stacking=4, max_steps=4500):
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = RawScoreEnv(env, max_steps)

    return env

class RawScoreEnv(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)

        self.steps = 0
        self.max_steps = max_steps

        self.raw_episodes = 0
        self.raw_score = 0.0
        self.raw_score_per_episode = 0.0
        self.raw_score_total = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['raw_score'] = reward

        self.steps += 1
        if self.steps >= self.max_steps:
            self.steps = 0
            done = True

        self.raw_score += reward
        self.raw_score_total += reward
        if done:
            self.steps = 0
            self.raw_episodes += 1

            k = 0.1
            self.raw_score_per_episode = (1.0 - k) * self.raw_score_per_episode + k * self.raw_score
            self.raw_score = 0.0

        reward = max(0., float(numpy.sign(reward)))

        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()

class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if numpy.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()
    
class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height=96, width=96, frame_stacking=4):
        super(ResizeEnv, self).__init__(env)
        self.height = height
        self.width = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)
        self.dtype = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.height, self.width))

        for i in reversed(range(self.frame_stacking - 1)):
            self.state[i + 1] = self.state[i].copy()
        self.state[0] = (numpy.array(img).astype(self.dtype) / 255.0).copy()

        return self.state
    
class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = numpy.zeros((2,) + self.env.observation_space.shape, dtype=numpy.uint8)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info

class MultiEnvParallel:
    def __init__(self, envs_list, envs_count, thread_count=4):
        dummy_env = envs_list[0]

        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

        self.envs_list = envs_list
        self.envs_count = envs_count
        self.threads_count = thread_count
        self.envs_per_thread = envs_count // thread_count

        self.observations = numpy.zeros((envs_count,) + self.observation_space.shape, dtype=numpy.float32)
        # a = [None] * n_env
        self.rewards = numpy.zeros((envs_count, 1), dtype=numpy.float32)
        self.dones = numpy.zeros((envs_count, 1), dtype=numpy.float32)
        self.infos = [None] * envs_count

        print("MultiEnvParallel")
        print("envs_count      = ", self.envs_count)
        print("threads_count   = ", self.threads_count)
        print("envs_per_thread = ", self.envs_per_thread)
        print("\n\n")

    def close(self):
        for i in range(self.envs_count):
            self.envs_list[i].close()

    def reset(self, env_id):
        return self.envs_list[env_id].reset()

    def render(self, env_id):
        pass

    def _step(self, param):
        index, action = param
        obs, reward, done, info = self.envs_list[index].step(action)

        self.observations[index] = obs
        self.rewards[index] = reward
        self.dones[index] = done
        self.infos[index] = info

    def step(self, actions):
        p = [(i, a) for i, a in zip(range(self.envs_count), actions)]
        with ThreadPoolExecutor(max_workers=self.threads_count) as executor:
            executor.map(self._step, p)

        obs = self.observations
        reward = self.rewards
        done = self.dones
        info = {}
        for i in self.infos:
            if i is not None:
                for k in i:
                    if k not in info:
                        info[k] = []
                    info[k].append(i[k])

        for k in info:
            info[k] = numpy.stack(info[k])

        return obs, reward, done, info

class ResultCollector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object ResultCollector')
            cls._instance = super(ResultCollector, cls).__new__(cls)
            cls._instance.collector = GenericCollector()
            cls._instance.global_step = 0
            cls._instance.n_env = 0

            cls._instance.collector_values = {}
            cls._instance.global_values = {}
        return cls._instance

    def init(self, n_env, **kwargs):
        self.collector.init(n_env, **kwargs)
        self.n_env = n_env

        for k in self.collector.keys:
            self.collector_values[k] = []

    def add(self, **kwargs):
        for k in kwargs:
            if k not in self.collector_values:
                self.collector.add(k, kwargs[k])
                self.collector_values[k] = []

    def update(self, **kwargs):
        self.collector.update(**kwargs)

        for k in [item for item in kwargs.keys() if item not in self.collector.keys]:
            if k not in self.global_values:
                self.global_values[k] = {}
            if self.global_step not in self.global_values[k]:
                self.global_values[k][self.global_step] = []
            self.global_values[k][self.global_step].append(kwargs[k].cpu().item())

    def reset(self, indices):
        result = None
        if len(indices) > 0:
            result = self.collector.reset(indices)

            for k in self.collector.keys:
                self.collector_values[k].append((result[k].step, result[k].sum, result[k].max, result[k].mean, result[k].std))

        return result

    def end_step(self):
        self.global_step += self.n_env

    def finalize(self):
        data = {}

        for k in self.collector_values.keys():
            data[k] = self._finalize_value(self.collector_values[k], ['step', 'sum', 'max', 'mean', 'std'], mode='cumsum_step')

        for k in self.global_values.keys():
            data[k] = self._finalize_value(self.global_values[k], ['step', 'val'], mode='mean_step')

        return data

    def clear(self):
        self.collector.clear()
        self.global_step = 0
        self.n_env = 0

        for k in self.collector_values.keys():
            self.collector_values[k].clear()
        self.collector_values.clear()

        for k in self.global_values.keys():
            self.global_values[k].clear()
        self.global_values.clear()

    @staticmethod
    def _finalize_value(value, keys, mode):
        result = {}

        if mode == 'cumsum_step':
            l = tuple(map(list, zip(*value)))
            for i, k in enumerate(keys):
                result[k] = np.array(list(itertools.chain(*l[i])))

            result['step'] = np.cumsum(result['step'])

        if mode == 'mean_step':
            l = []
            for steps in value:
                val = sum(value[steps]) / len(value[steps])
                l.append((steps, val))

            l = tuple(map(list, zip(*l)))
            for i, k in enumerate(keys):
                result[k] = np.array(l[i])

        return result
    
class GenericCollector:
    def __init__(self):
        self.keys = []
        self._n_env = 0
        self._buffer = {}

        self._simple_stats = namedtuple('simple_stats', ['step', 'max', 'sum', 'mean', 'std'])

    def init(self, n_env, **kwargs):
        self._n_env = n_env
        for k in kwargs:
            self.add(k, kwargs[k])

    def add(self, key, shape):
        self.keys.append(key)
        self._buffer[key] = RunningStats(shape, 'cpu', n=self._n_env)

    def update(self, **kwargs):
        for k in kwargs:
            if k in self._buffer:
                self._buffer[k].update(kwargs[k], reduction='none')

    def reset(self, indices):
        result = {}

        for k in self._buffer:
            result[k] = []
            for i in indices:
                result[k].append(self._evaluate(k, i))
                self._buffer[k].reset(i)
            result[k] = self._simple_stats(*tuple(map(list, zip(*result[k]))))

        return result

    def _evaluate(self, key, index):
        return [self._buffer[key].count[index].item() - 1, self._buffer[key].max[index].item(), self._buffer[key].sum[index].item(), self._buffer[key].mean[index].item(), self._buffer[key].std[index].item()]

    def clear(self):
        self.keys.clear()
        self._n_env = 0
        self._buffer = {}

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