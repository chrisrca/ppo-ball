from learner import Learner

def build_ball_game_env():
    from ball_game_env import BallGameEnv
    return BallGameEnv()

if __name__ == "__main__":
    n_proc = 96
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_ball_game_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=True,
                      wandb_project_name="ppo-ball")
    learner.learn()