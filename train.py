import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agent import Agent
from src.config import get_agent_config, get_training_config
from src.envs import make_env
from src.evaluation import evaluate_agent


def plot_final_curve(episode_returns, eval_points, eval_means, eval_stds, figure_path):
    plt.figure(figsize=(10, 5))

    plt.plot(episode_returns, label="Training return", alpha=0.35)

    if len(episode_returns) >= 10:
        moving_average = np.convolve(
            episode_returns,
            np.ones(10) / 10,
            mode="valid",
        )
        plt.plot(
            range(10, len(episode_returns) + 1),
            moving_average,
            label="10-episode training average",
        )

    if eval_points:
        plt.errorbar(
            eval_points,
            eval_means,
            yerr=eval_stds,
            fmt="o-",
            label="Evaluation mean ± std",
            capsize=4,
        )

    plt.axhline(y=300, linestyle="--", label="Solved threshold")
    plt.xlabel("Training episode")
    plt.ylabel("Return")
    plt.title("SAC BipedalWalker Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardcore", action="store_true")
    args = parser.parse_args()

    mode = "sac_hardcore" if args.hardcore else "sac_normal"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent_config = get_agent_config(args.hardcore)
    training_config = get_training_config(args.hardcore)

    max_episodes = training_config["max_episodes"]
    eval_every = training_config["eval_every"]
    eval_episodes = training_config["eval_episodes"]

    env = make_env(hardcore=args.hardcore, render_mode=None)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    agent = Agent(env, **agent_config)

    log_dir = Path("logs") / mode
    model_dir = Path("models") / mode
    video_dir = Path("videos") / mode
    figure_dir = Path("docs") / "figures" / mode

    for directory in [log_dir, model_dir, video_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_log_path = log_dir / "training_log.csv"
    eval_log_path = log_dir / "evaluation_log.csv"
    figure_path = figure_dir / "training_curve.png"

    episode_returns = []
    eval_points = []
    eval_means = []
    eval_stds = []

    best_eval_return = -float("inf")

    print(f"Mode: {mode}")
    print(f"Agent config: {agent_config}")
    print(f"Training config: {training_config}")
    print(f"Logs:   {log_dir}")
    print(f"Models: {model_dir}")
    print(f"Videos: {video_dir}")

    with open(train_log_path, "w", newline="") as train_file, open(
        eval_log_path, "w", newline=""
    ) as eval_file:
        train_writer = csv.writer(train_file)
        eval_writer = csv.writer(eval_file)

        train_writer.writerow(["episode", "return", "length", "mean_return_last_10"])
        eval_writer.writerow(
            [
                "episode",
                "mean_return",
                "std_return",
                "min_return",
                "max_return",
                "returns",
            ]
        )

        for episode_index in range(max_episodes):
            episode_number = episode_index + 1

            episode_info = agent.episode()
            episode_return = episode_info["episode_return"]
            episode_length = episode_info["episode_length"]

            episode_returns.append(episode_return)
            mean_return_last_10 = float(np.mean(episode_returns[-10:]))

            train_writer.writerow(
                [episode_number, episode_return, episode_length, mean_return_last_10]
            )
            train_file.flush()

            if episode_number % 10 == 0:
                print(
                    f"Episode {episode_number:4d} | "
                    f"Mean Return (last 10): {mean_return_last_10:8.2f} | "
                    f"Steps: {agent.total_steps}"
                )

            if episode_number % eval_every == 0:
                eval_result = evaluate_agent(
                    agent=agent,
                    episodes=eval_episodes,
                    seeds=[1000 + episode_number + i for i in range(eval_episodes)],
                    hardcore=args.hardcore,
                    record_video=True,
                    video_folder=str(video_dir),
                    video_prefix=f"{mode}_eval_ep_{episode_number}",
                )

                eval_points.append(episode_number)
                eval_means.append(eval_result["mean_return"])
                eval_stds.append(eval_result["std_return"])

                eval_writer.writerow(
                    [
                        episode_number,
                        eval_result["mean_return"],
                        eval_result["std_return"],
                        eval_result["min_return"],
                        eval_result["max_return"],
                        eval_result["returns"],
                    ]
                )
                eval_file.flush()

                agent.save(model_dir / f"checkpoint_ep_{episode_number}.pt")

                if eval_result["mean_return"] > best_eval_return:
                    best_eval_return = eval_result["mean_return"]
                    agent.save(model_dir / "best.pt")
                    print(f"New best model saved: {best_eval_return:.2f}")

                print(
                    f"[Eval @ Episode {episode_number}] "
                    f"Mean: {eval_result['mean_return']:.2f} | "
                    f"Std: {eval_result['std_return']:.2f} | "
                    f"Min: {eval_result['min_return']:.2f} | "
                    f"Max: {eval_result['max_return']:.2f}"
                )

    env.close()

    final_model_path = model_dir / "final.pt"
    agent.save(final_model_path)

    plot_final_curve(
        episode_returns=episode_returns,
        eval_points=eval_points,
        eval_means=eval_means,
        eval_stds=eval_stds,
        figure_path=figure_path,
    )

    print(f"Saved training log to: {train_log_path}")
    print(f"Saved evaluation log to: {eval_log_path}")
    print(f"Saved training curve to: {figure_path}")
    print(f"Saved final model to: {final_model_path}")
    print(f"Best eval return: {best_eval_return:.2f}")


if __name__ == "__main__":
    main()
