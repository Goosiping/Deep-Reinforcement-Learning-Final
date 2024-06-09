import gym
import cv2
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from utils import *
from PPOAgent import Agent
import pandas as pd

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

env = WrapperAtari(gym.make('MontezumaRevengeNoFrameskip-v4'))
input_shape = env.observation_space.shape
action_dim = env.action_space.n
print(input_shape)
print(action_dim)

agent = Agent((4, 96, 96), 
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
agent.load('./models/FinalModel')

video_path = 'rnd_.mp4'
video_recorder = VideoRecorder(env, video_path)
state0 = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(device)
done = False
score = 0
steps = 0
save_skip = 4
print(state0.shape)

while not done:
    video_recorder.capture_frame()
    # features0 = agent.get_features(state0)
    # _, _, action0, probs0 = agent.get_action(state0)
    _, action0, probs0 = agent.get_action(state0)
    # actor_state, value, action0, probs0, head_value, head_action, head_probs, all_values, all_action, all_probs = agent.get_action(state0)
    # action0 = probs0.argmax(dim=1)
    next_state, reward, done, info = env.step(agent.convert_action(action0.cpu())[0])
    state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
    score += info['raw_score']
    steps += 1
    # print(info)

    if(steps % save_skip == 0):
        predicted_code, target_code = agent.network.rnd_model(state0)
        
        if 'df' in locals():
            df = pd.concat([df, pd.DataFrame(target_code.cpu().detach().numpy())])
            print(target_code.cpu().detach().numpy()[0, :6])
            cv2.imwrite('./rooms_frames/{0}.png'.format(steps), next_state[-1,:,:] * 255)
        else:
            df = pd.DataFrame(target_code.cpu().detach().numpy())
            print(target_code.cpu().detach().numpy()[0, :6])
            cv2.imwrite('./rooms_frames/{0}.png'.format(steps), next_state[-1,:,:] * 255)

    cv2.imshow('frame', next_state[-1,:,:])
    cv2.waitKey(1)
    # print(reward)
print(score)
print(steps)
df.to_csv('embeddings.csv')
video_recorder.close()
cv2.waitKey(0)
cv2.destroyAllWindows()