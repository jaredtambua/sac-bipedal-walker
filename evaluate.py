import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.agent import Agent
from src.config import get_agent_config
from src.envs import make_env
from src.evaluation import evaluate_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardcore", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    mode = "sac_hardcore" if args.hardcore else "sac_normal"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_path = (
        Path(args.model)
        if args.model is not None
        else Path("models") / mode / "best.pt"
    )

    video_dir = Path("videos") / mode / "manual_eval"
    video_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(hardcore=args.hardcore, render_mode=None)

    agent = Agent(
        env,
        **get_agent_config(args.hardcore),
    )

    agent.load(model_path)

    result = evaluate_agent(
        agent=agent,
        episodes=args.episodes,
        seeds=[4000 + i for i in range(args.episodes)],
        hardcore=args.hardcore,
        record_video=not args.no_video,
        video_folder=str(video_dir),
        video_prefix=f"{mode}_manual_eval",
    )

    env.close()

    print(f"Loaded model from: {model_path}")
    print(f"Mean return: {result['mean_return']:.2f}")
    print(f"Std return:  {result['std_return']:.2f}")
    print(f"Min return:  {result['min_return']:.2f}")
    print(f"Max return:  {result['max_return']:.2f}")
    print(f"Returns:     {result['returns']}")

    if not args.no_video:
        print(f"Saved videos to: {video_dir}")


if __name__ == "__main__":
    main()
