import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils.environment import CustomMsPacmanEnv
from utils.wrappers import MaxAndSkipEnv, ProcessFrame84Gray, BufferWrapper
from utils.agent import Agent
from utils.replay_buffer import ExperienceBuffer
from utils.loss import calc_loss
from models.dqn_model import DQN
from models.resnet_dqn_model import ResNetDQN
import datetime
import argparse

def train(model_name):
    env = CustomMsPacmanEnv(".\ms_pacman.bin")
    legal_actions = env.legal_actions
    env = MaxAndSkipEnv(env, skip=16)
    env = ProcessFrame84Gray(env)
    env = BufferWrapper(env, n_steps=1,legal_actions=env.legal_actions)
    input_shape = (1, 84, 84)
    n_actions = len(env.legal_actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "DQN":
        net = DQN(input_shape, n_actions).to(device)
        tgt_net = DQN(input_shape, n_actions).to(device)
    elif model_name == "resnet":
        net = ResNetDQN((1,84,84), len(legal_actions)).to(device)
        tgt_net = ResNetDQN((1,84,84), len(legal_actions)).to(device)
    else:
        raise ValueError("Unknown model name")

    buffer = ExperienceBuffer(10000)
    agent = Agent(env, buffer)

    epsilon_start = 1.0
    epsilon_decay = 0.99
    min_epsilon = 0.01
    learning_rate = 1e-4
    batch_size = 128
    sync_target_frames = 1000
    epsilon_decay_last_frame=250000

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    total_rewards = []
    frame_idx = 0
    best_m_reward = None
    history = []

    while frame_idx < 2000000:
        frame_idx += 1
        epsilon = max(min_epsilon, epsilon_start - frame_idx / epsilon_decay_last_frame)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            history.append((frame_idx, m_reward, epsilon))
            if best_m_reward is None or best_m_reward < m_reward:
                # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"best_model_{model_name}_{batch_size}.dat"
                torch.save(net.state_dict(), model_filename)
                if best_m_reward is not None:
                    print(f"{frame_idx}: best reward updated {best_m_reward:.3f} -> {m_reward:.3f} (eps: {epsilon:.2})")
                best_m_reward = m_reward

        if len(buffer) < 10000:
            continue

        if frame_idx % sync_target_frames == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    history_array = np.array(history)
    plt.plot(history_array[:, 0], history_array[:, 1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False,default="resnet", choices=["DQN", "resnet"], help="Choose the model to train")
    args = parser.parse_args()
    train(args.model)