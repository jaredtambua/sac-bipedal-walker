def get_agent_config(hardcore: bool):
    if hardcore:
        return {
            "seed": 42,
            "discount": 0.99,
            "tau": 0.005,
            "buffer_size": 1000000,
            "batch_size": 256,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "initial_alpha": 0.2,
            "warmup_steps": 20000,
            "updates_per_step": 1,
            "hidden": 256,
        }

    return {
        "seed": 42,
        "discount": 0.99,
        "tau": 0.005,
        "buffer_size": 1000000,
        "batch_size": 256,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "initial_alpha": 0.2,
        "warmup_steps": 10000,
        "updates_per_step": 1,
        "hidden": 256,
    }


def get_training_config(hardcore: bool):
    if hardcore:
        return {
            "max_episodes": 8000,
            "eval_every": 500,
            "eval_episodes": 3,
        }

    return {
        "max_episodes": 1000,
        "eval_every": 100,
        "eval_episodes": 5,
    }
