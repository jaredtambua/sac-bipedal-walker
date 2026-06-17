import gymnasium as gym


def make_env(hardcore=False, render_mode=None):
    return gym.make(
        "BipedalWalker-v3",
        hardcore=hardcore,
        render_mode=render_mode,
    )
