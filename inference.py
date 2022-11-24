import os
import numpy as np
import torch
import gym
from helper import select_action, show_video
import skvideo.io

def test(path, env_name):
    frames = []
    model = torch.load(path+'model_final.pt')
    model.eval()

    env = gym.make(env_name)
    state = env.reset()
    for _ in range(200):
        action = select_action(model, env, state)
        state, reward, done, _ = env.step(action)
        frame = env.sim.render(width=500, height=500,mode='offscreen', camera_name='hand_side_inter')
        frames.append(frame[::-1,:,:])
    env.close()
    # make a local copy
    skvideo.io.vwrite('videos/{}-model.mp4'.format(env_name), np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})

    # show in the notebook
    show_video('videos/{}-model.mp4'.format(env_name))