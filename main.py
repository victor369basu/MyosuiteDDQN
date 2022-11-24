import torch
import numpy as np
import random
import torch.nn.functional as F
import gym
import myosuite
import os
from model import QNetwork
from helper import update_parameters, Memory, select_action
from evaluation import evaluate
from train import train
from inference import test
from config import get_parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main(gamma=0.99,
         lr=3e-5, 
         min_episodes=3,
         eps=0.09,
         batch_size=64, 
         update_repeats=50,
         num_episodes=700, 
         seed=0, 
         max_memory_size=50000,
         measure_step=10,
         measure_repeats=100, 
         hidden_dim=128,
         d_model=256,
         ntoken=256,
         nhead=8,
         nlayers=2,
         dropout=0.0,
         horizon=np.inf,
         path='/content/',
         loss_fn = 'mse', 
         env_name="myoHandReachFixed-v0"):
    env = gym.make(env_name)
    env.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    Q_1 = QNetwork(d_model=d_model, ntoken=ntoken,
                    action_dim=env.action_space.shape[0], 
                    nhead=nhead, nlayers=nlayers, dropout=dropout,
                    state_dim=env.observation_space.shape[0],
                    hidden_dim=hidden_dim).to(device)
    Q_2 = QNetwork(d_model=d_model, ntoken=ntoken,
                    action_dim=env.action_space.shape[0], 
                    nhead=nhead, nlayers=nlayers, dropout=dropout,
                    state_dim=env.observation_space.shape[0],
                    hidden_dim=hidden_dim).to(device)
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)
    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.RMSprop(Q_1.parameters(), lr=lr, eps=eps)
    memory = Memory(max_memory_size)
    performance = []
    best_performance = -np.inf

    for episode in range(num_episodes+1):
        # display the performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(Q_1, env, measure_repeats, device)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            if performance[-1][1] > best_performance:
                torch.save(Q_1, path + 'model_best.pt')
                best_performance = performance[-1][1]

        state = env.reset()
        memory.state.append(state)

        done = False
        i = 0
        while not done:
            i += 1
            action = select_action(Q_2, env, state)
            state, reward, done, _ = env.step(action)

            if i > horizon:
                done = True

            # save state, action, reward sequence
            memory.update(state, action, reward, done)

        if episode >= min_episodes : #and episode % update_step == 0:
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma, loss_fn,device)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

    return Q_1, performance

if __name__ == '__main__':
    config = get_parameters()
    print("Model Configuration: ")
    print(config)
    # Load the data in the appropriate format for training
    if config.train:
        os.makedirs(config.path, exist_ok=True)
        Q_1, performance = main(gamma=config.gamma,
            lr=config.learning_rate, 
            min_episodes=config.min_episodes,
            eps=config.eps,
            batch_size=config.batch_size, 
            update_repeats=config.update_repeats,
            num_episodes=config.epochs, 
            seed=config.seed, 
            max_memory_size=config.max_memory_size,
            measure_step=config.measure_step,
            measure_repeats=config.measure_repeats, 
            hidden_dim=config.hidden_dim,
            d_model=config.d_model,
            ntoken=config.ntoken,
            nhead=config.nhead,
            nlayers=config.nlayers,
            dropout=config.dropout,
            horizon=np.inf,
            path=config.path,
            loss_fn = config.loss_fn, 
            env_name=config.env_name)
        torch.save(Q_1, config.path+'model_final.pt')
    else:
        os.makedirs('videos', exist_ok=True)
        test(config.path, config.env_name)