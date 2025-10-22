import pygame
from stable_baselines3 import PPO
from ball_game_env import BallGameEnv
import time

# Create environment with rendering
env = BallGameEnv(render_mode='human')
# Load the trained model
model = PPO.load("ppo_ball_model")
# Reset environment
obs, _ = env.reset()
episode_count = 0
total_reward = 0

# Run the model
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode {episode_count + 1} reward: {total_reward:.2f}")
        obs, _ = env.reset()
        total_reward = 0
        episode_count += 1
        time.sleep(1)
    time.sleep(0.016)

# Close the environment
env.close()