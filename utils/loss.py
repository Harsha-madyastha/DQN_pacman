import torch
import torch.nn as nn
import numpy as np

def calc_loss(batch, net, tgt_net, device, gamma=0.99):
    states, actions, rewards, dones, next_states = batch

    states = np.array(states).reshape(-1,1,84,84)
    next_states = np.array(next_states).reshape(-1,1,84,84)

    states_v = torch.tensor(states).to(device, dtype=torch.float32)
    next_states_v = torch.tensor(next_states).to(device, dtype=torch.float32)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)
