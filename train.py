from etaprogress.progress import ProgressBar
from PPOAgent import Agent
from utils import *
import gym
import torch
import numpy
import wandb

# Parameters
device = 'cuda'
checkpoint = 10000
n_env = 128
trials = 9
steps = 128
gamma = "0.998,0.99"   #0.99
beta = 0.001
batch_size = 512
trajectory_size = 16384
ppo_epochs = 4
lr = 0.0001
actor_loss_weight = 1
critic_loss_weight = 0.5
motivation_lr = 0.0001
motivation_eta = 1
num_threads = 4

class StepCounter:
    def __init__(self, limit):
        self.limit = limit
        self.steps = 0
        self.bar = ProgressBar(limit, max_width=40)

    def update(self, value):
        self.steps += value
        if self.steps > self.limit:
            self.steps = self.limit
        self.bar.numerator = self.steps

    def print(self):
        print(self.bar)

    def running(self):
        return self.steps < self.limit

if __name__ == "__main__":

    # Environment
    env_name = 'MontezumaRevengeNoFrameskip-v4'
    env = MultiEnvParallel([WrapperHardAtari(gym.make(env_name)) for _ in range(n_env)], n_env, num_threads)
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # Agent
    agent = Agent(input_shape, 
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
                  motivation_eta,)
    
    # Setups some tools
    experiment = wandb.init(project='RND', 
                            config={'env': env_name, 'n_env': n_env, 'trials': trials, 'steps': steps, 'gamma': gamma, 'beta': beta, 'batch_size': batch_size, 'trajectory_size': trajectory_size, 'ppo_epochs': ppo_epochs, 'lr': lr, 'actor_loss_weight': actor_loss_weight, 'critic_loss_weight': critic_loss_weight, 'motivation_lr': motivation_lr, 'motivation_eta': motivation_eta, 'num_threads': num_threads})
    step_counter = StepCounter(int(steps * 1e6))
    analytic = ResultCollector()
    analytic.init(n_env, re=(1,), score=(1,), ri=(1,), error=(1,), ext_value=(1,), int_value=(1,))
    reward_avg = RunningAverageWindow(100)
    time_estimator = PPOTimeEstimator(step_counter.limit)
    s = numpy.zeros((n_env,) + env.observation_space.shape, dtype=numpy.float32)
    for i in range(n_env):
        s[i] = env.reset(i)
    state0 = agent.process_state(s)

    # Start training
    while step_counter.running():
        agent.motivation.update_state_average(state0)
        with torch.no_grad():
            value, action0, probs0 = agent.get_action(state0)
        next_state, reward, done, info = env.step(agent.convert_action(action0.cpu()))

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        state1 = agent.process_state(next_state)
        int_reward = agent.motivation.reward(state1).cpu().clip(0.0, 1.0)

        # Process data
        if info is not None:
            if 'normalised_score' in info:
                analytic.add(normalised_score=(1,))
                score = torch.tensor(info['normalised_score']).unsqueeze(-1)
                analytic.update(normalised_score=score)
            if 'raw_score' in info:
                analytic.add(score=(1,))
                score = torch.tensor(info['raw_score']).unsqueeze(-1)
                analytic.update(score=score)
        error = agent.motivation.error(state0).cpu()
        analytic.update(re=ext_reward,
                            ri=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error)
        
        env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
        stats = analytic.reset(env_indices)
        step_counter.update(n_env)

        # print some results and wirte to log
        for i, index in enumerate(env_indices):
            reward_avg.update(stats['re'].sum[i])

            print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f})]'.format(
                1, step_counter.steps, step_counter.limit, stats['re'].sum[i], stats['ri'].max[i], stats['ri'].mean[i],
                stats['ri'].std[i],
                int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i]))
            # print(type(step_counter.steps), type(step_counter.limit), type(stats['re'].sum[i]), type(stats['ri'].max[i]), type(stats['ri'].mean[i]), type(stats['ri'].std[i]), type(int(stats['re'].step[i])), type(reward_avg.value().item()), type(stats['score'].sum[i]))
            with open('log.txt', 'a') as log:
                log.write('step {0:d}/{1:d} ex_reward {2:f} int_reward_max {3:f} int_reward_mean {4:f} int_reward_std {5:f} steps {6:d} mean_reward {7:f} score {8:f}\n'.format(
                    step_counter.steps, step_counter.limit, stats['re'].sum[i], stats['ri'].max[i], stats['ri'].mean[i],stats['ri'].std[i],int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i]
                ))
            print(time_estimator)
            next_state[i] = env.reset(index)

            experiment.log({'ext_reward': stats['re'].sum[i],
                            'int_reward_max': stats['ri'].max[i],
                            'int_reward_mean': stats['ri'].mean[i],
                            'int_reward_std': stats['ri'].std[i],
                            'mean_reward': reward_avg.value().item(),
                            'score': stats['score'].sum[i]})


        reward = torch.cat([ext_reward, int_reward], dim=1)
        done = torch.tensor(1 - done, dtype=torch.float32)
        analytic.end_step()

        agent.train(state0, value, action0, probs0, state1, reward, done)

        state0 = state1
        time_estimator.update(n_env)

        # TODO: Checkpoint
        if step_counter.steps % checkpoint == 0:
            agent.save('./models/Model_{0}'.format(step_counter.steps))

    

    # Training finish
    agent.save('./models/FinalModel')

    print('Saving data...')
    analytic.reset(numpy.array(range(n_env)))
    save_data = analytic.finalize()
    numpy.save('ppo_{0}'.format(1), save_data)
    analytic.clear()
