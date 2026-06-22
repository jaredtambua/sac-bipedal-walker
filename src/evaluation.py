import numpy as np
from gymnasium.wrappers import RecordVideo

from src.envs import make_env


def evaluate_agent(
    agent,
    episodes=100,
    seeds=None,
    hardcore=False,
    record_video=True,
    video_folder="videos",
    video_prefix="eval",
):
    if seeds is None:
        seeds = list(range(episodes))

    eval_env = make_env(
        hardcore=hardcore,
        render_mode="rgb_array" if record_video else None,
    )

    if record_video:
        eval_env = RecordVideo(
            eval_env,
            video_folder=video_folder,
            name_prefix=video_prefix,
            episode_trigger=lambda episode_id: True,
        )

    returns = []

    for seed in seeds:
        observation, info = eval_env.reset(seed=seed)
        eval_env.action_space.seed(seed)

        done = False
        total_reward = 0.0

        while not done:
            action = agent.algo.sample_action(
                observation,
                explore=False,
            )

            action = np.asarray(action, dtype=np.float32)
            action = np.clip(
                action,
                eval_env.action_space.low.astype(np.float32),
                eval_env.action_space.high.astype(np.float32),
            ).astype(np.float32)

            observation, reward, terminated, truncated, info = eval_env.step(action)

            total_reward += float(reward)
            done = bool(terminated or truncated)

        returns.append(total_reward)

    eval_env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "returns": returns,
    }
