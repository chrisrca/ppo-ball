from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from ball_game_env import BallGameEnv

def make_env():
    return BallGameEnv()

if __name__ == '__main__':
    n_envs = 16
    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    policy_kwargs = {
        'net_arch': {
            'pi': [256, 256], # Actor
            'vf': [256, 256] # Critic
        }
    }
    model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu', policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=1000000)
    model.save("ppo_ball_model")
    vec_env.close()