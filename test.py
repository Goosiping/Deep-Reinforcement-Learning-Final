import gym
import cv2

env_name = 'MontezumaRevengeNoFrameskip-v4'
# (210, 160, 3)
# Discard(18)
env = gym.make(env_name)

env.reset()
print(env.action_space)
for _ in range(10000):
    s, r, d, i= env.step(env.action_space.sample())

    s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', s)    
    cv2.waitKey(1)
    while d:
        break
