from torch import nn
import torch
import torch.nn.functional as F

def evaluate(Qmodel, env, repeats, device):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = values.cpu().numpy()
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform/repeats