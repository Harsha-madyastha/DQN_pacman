import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import cv2
from ale_py import ALEInterface
from random import randrange

# Custom environment class
class CustomMsPacmanEnv:
    def __init__(self, rom_path):
        self.ale = ALEInterface()
        self.ale.loadROM(rom_path)
        self.legal_actions = self.ale.getLegalActionSet()
        self.reset()

    def reset(self):
        self.ale.reset_game()
        return self.get_screen()

    def step(self, action):
        reward = self.ale.act(action)
        screen_obs = self.get_screen()
        done = self.ale.game_over()
        return screen_obs, reward, done, None

    def get_screen(self):
        screen = self.ale.getScreenRGB()
        return screen

# Max and Skip Wrapper
class MaxAndSkipEnv:
    def __init__(self, env, skip=16):
        self.env = env
        self._skip = skip
        self._obs_buffer = collections.deque(maxlen=2)
        self.legal_actions = env.legal_actions

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)

# Process Frame Wrapper
class ProcessFrame84:
    def __init__(self, env):
        self.env = env
        self.legal_actions = env.legal_actions

    def reset(self):
        return self.process(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.process(obs), reward, done, info

    @staticmethod
    def process(frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.229 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [84, 84, 1])
        return x_t.astype(np.uint8)
    
    def __getattr__(self, name):
        return getattr(self.env, name)

# Buffer Wrapper
class BufferWrapper:
    def __init__(self, env, n_steps,legal_actions, dtype=np.float32):
        self.env = env
        self.dtype = dtype
        self.n_steps = n_steps
        self.legal_actions = legal_actions
        self.buffer = None

    def reset(self):
        self.buffer = np.zeros((self.n_steps, 84, 84, 1), dtype=self.dtype)
        return self.observation(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return torch.tensor(np.transpose(self.buffer, (0, 3, 1, 2)), dtype=torch.float32)
    
    def __getattr__(self, name):
        return getattr(self.env, name)

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        conv_out = self.conv(x.view(batch_size, -1, x.size(3), x.size(4)))  # Reshape for convolutional layers
        conv_out = conv_out.view(batch_size, -1)  # Flatten for fully connected layers
        return self.fc(conv_out)

# Experience Buffer
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), \
               np.array(next_states)

# Agent
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        if np.random.random() < epsilon:
            action = randrange(len(self.env.legal_actions))
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device, dtype=torch.float32)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device):
    GAMMA = 0.99
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device, dtype=torch.float32)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device, dtype=torch.float32)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

# Training Loop
def train():
    env = CustomMsPacmanEnv("E:\\dqn-pacman\\venv\\Lib\\site-packages\\ale_py\\roms\\ms_pacman.bin")
    model_name="DQN"
    batch_size=128
    legal_actions = env.legal_actions
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    env = BufferWrapper(env, n_steps=4,legal_actions=legal_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DQN((4, 84, 84), len(legal_actions)).to(device)
    tgt_net = DQN((4, 84, 84), len(legal_actions)).to(device)
    buffer = ExperienceBuffer(10000)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    total_rewards = []
    frame_idx = 0
    best_m_reward = None

    while frame_idx < 2000000:
        frame_idx += 1
        epsilon = max(0.01, 1.0 - frame_idx / 250000)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), f"best_model_{model_name}_{batch_size}.dat")
                best_m_reward = m_reward
            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, eps {epsilon:.2f}")

        if len(buffer) < 10000:
            continue

        if frame_idx % 1000 == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

if __name__ == "__main__":
    train()
