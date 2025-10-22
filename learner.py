from stable_baselines3 import PPO
from ball_game_env import BallGameEnv

env = BallGameEnv()
model = PPO("MlpPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=1000000)
model.save("ppo_ball_model")
env.close()