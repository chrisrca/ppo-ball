import pygame
import os
import torch
from ball_game_env import BallGameEnv
import time
from rlgym_ppo.ppo.discrete_policy import DiscreteFF

def build_ball_game_env():
    return BallGameEnv(render_mode='human')

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Select the latest checkpoint
    checkpoints_base = "data/checkpoints"
    latest_run = None
    latest_timestamp = -1
    for run_folder in os.listdir(checkpoints_base):
        if run_folder.startswith("rlgym-ppo-run-") and run_folder.split("-")[-1].isdigit():
            timestamp = int(run_folder.split("-")[-1])
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_run = run_folder
    if latest_run is None:
        raise FileNotFoundError("No checkpoint folders found in data/checkpoints")
    run_path = os.path.join(checkpoints_base, latest_run)
    latest_ts = -1
    for ts_folder in os.listdir(run_path):
        if ts_folder.isdigit() and int(ts_folder) > latest_ts:
            latest_ts = int(ts_folder)
    if latest_ts == -1:
        raise FileNotFoundError(f"No timestep folders found in {run_path}")
    checkpoint_folder = os.path.join(run_path, str(latest_ts))
    print(f"Loading policy from: {checkpoint_folder}")

    # Policy parameters
    obs_space_size = 6
    n_actions = 3
    policy_layer_sizes = (256, 256, 256)
    policy = DiscreteFF(obs_space_size, n_actions, policy_layer_sizes, device)
    policy_path = os.path.join(checkpoint_folder, "PPO_POLICY.pt")
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    # Create environment with rendering
    env = BallGameEnv(render_mode='human')

    # Reset environment
    obs = env.reset()
    episode_count = 0
    total_reward = 0

    # Run the model
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
        
        with torch.no_grad():
            action, _ = policy.get_action(obs, deterministic=False)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode {episode_count + 1} reward: {total_reward:.2f}")
            obs = env.reset()
            total_reward = 0
            episode_count += 1
            time.sleep(1)
        time.sleep(0.016)