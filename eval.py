import torch
from utils.environment import CustomMsPacmanEnv
from utils.wrappers import MaxAndSkipEnv, ProcessFrame84Gray, BufferWrapper
from models.dqn_model import DQN
from models.resnet_dqn_model import ResNetDQN
import numpy as np
import imageio
import os

def evaluate(model_path, model_type, batch_size):
    """
    Evaluates a reinforcement learning model on the Ms. Pacman environment and saves an animation as a GIF.

    Args:
        model_path (str): Path to the model file to evaluate.
        model_type (str): Type of the model ('dqn' or 'resnet').
        batch_size (int): Batch size used during training of the model.

     Notes:
        - It saves the animation of the gameplay as a GIF in the './results' directory.
    """
    env = CustomMsPacmanEnv("./ms_pacman.bin")
    legal_actions = env.legal_actions
    env = MaxAndSkipEnv(env,skip=16)
    env = ProcessFrame84Gray(env)
    env = BufferWrapper(env, n_steps=1,legal_actions=env.legal_actions)
    input_shape = (1, 84, 84)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'resnet':
        net = ResNetDQN(input_shape, len(legal_actions)).to(device)
    else:
        net = DQN(input_shape, len(legal_actions)).to(device)
    
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    state = env.reset()
    total_reward = 0.0
    frames = []
    
    while True:
        frames.append(env.get_screen())
        state_v = torch.tensor(np.array(state).reshape(1, 1, 84, 84)).to(device, dtype=torch.float32)
        q_vals = net(state_v)
        _, action_v = torch.max(q_vals, dim=1)
        action = int(action_v.item())
        
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"Model: {model_path}, Total reward: {total_reward}")

    os.makedirs('./results', exist_ok=True)
    
    gif_path = f"./results/animation_{model_type}_{batch_size}.gif"
    imageio.mimsave(gif_path, frames, fps=10)

if __name__ == "__main__":
    model_types = ['DQN', 'resnet']
    batch_sizes = [32, 64, 128]
    
    for model_type in model_types:
        for batch_size in batch_sizes:
            model_file = f"best_model_{model_type}_{batch_size}.dat"
            evaluate(model_file, model_type.lower(), batch_size)
