# BipedalWalker Reinforcement Learning

A reinforcement learning project exploring continuous-control locomotion in the Gymnasium BipedalWalker environments.

The project trains agents to solve:

* BipedalWalker-v3
* BipedalWalkerHardcore-v3

using deep reinforcement learning and evaluates them across 100 unique terrain seeds.

---

## Results

| Environment              | Mean Return | Std Return | Episodes to Solve |
| ------------------------ | ----------: | ---------: | ----------------: |
| BipedalWalker-v3         |      310.20 |      38.14 |               300 |
| BipedalWalkerHardcore-v3 |      300.22 |      52.63 |              3000 |

Detailed evaluation results can be found in:

```text
docs/results.md
```

---

## Trained Models

The repository includes the earliest checkpoints that successfully solved each environment.

```text
models/normal/solved.pt
models/hardcore/solved.pt
```

These checkpoints can be evaluated directly using the provided evaluation script.

---

## Demonstration Videos

### Normal Environment

| Video | Return |
|--------|--------:|
| [Demo Run 1](videos/sac_normal/manual_eval/demo1.mp4) | 319.10 |
| [Demo Run 2](videos/sac_normal/manual_eval/demo2.mp4) | 317.96 |

### Hardcore Environment

| Video | Return |
|--------|--------:|
| [Demo Run 1](videos/sac_hardcore/manual_eval/demo1.mp4) | 307.46 |
| [Demo Run 2](videos/sac_hardcore/manual_eval/demo2.mp4) | 310.37 |
---

## Evaluation Protocol

Policies are evaluated:

* Deterministically (no exploration noise)
* Across 100 unique seeds
* Using the environment's original reward function

Metrics reported include:

* Mean Return
* Standard Deviation
* Minimum Return
* Maximum Return

---

## Running Training

Normal environment:

```bash
python train.py
```

Hardcore environment:

```bash
python train.py --hardcore
```

---

## Running Evaluation

Evaluate the normal solved model:

```bash
python evaluate.py --model models/normal/solved.pt --episodes 100
```

Evaluate the hardcore solved model:

```bash
python evaluate.py --hardcore --model models/hardcore/solved.pt --episodes 100
```

---

## Notes

This project focuses on sample-efficient locomotion learning and reproducible evaluation. The repository includes trained checkpoints, evaluation videos, training curves, and detailed performance results to allow independent verification of reported scores.
