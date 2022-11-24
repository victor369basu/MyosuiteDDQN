from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
total_loss=[]
loss_fn = nn.CrossEntropyLoss()

def train(batch_size, current, target, optim, memory, gamma, cost_fn, device):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    actions = torch.tensor(np.argmax(actions, axis=1)).to(device)
    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)
    

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)
    
    if cost_fn=='mse':
        loss = F.mse_loss(q_value, expected_q_value)
    if cost_fn=='cel':
        loss = loss_fn(q_value, expected_q_value)

    total_loss.append(loss)

    optim.zero_grad()
    loss.backward()
    optim.step()
