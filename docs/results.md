## Results

### Environment

The agent was trained and evaluated on:

* BipedalWalker-v3
* BipedalWalkerHardcore-v3

Evaluation was performed using deterministic policies over **100 unique seeds**.

---

## Normal Environment

### Solved Checkpoint

Model:

```text
models/sac_normal/ep_300.pt
```

Evaluation Results:

| Metric              | Value  |
| ------------------- | ------ |
| Mean Return         | 310.20 |
| Standard Deviation  | 38.14  |
| Minimum Return      | 78.74  |
| Maximum Return      | 320.97 |
| Evaluation Episodes | 100    |

#### Observations

* The policy consistently achieved returns around 315-320.
* A small number of evaluation seeds produced significantly lower returns.
* The average return remained above the Gymnasium solve threshold of 300.

---

## Hardcore Environment

### Solved Checkpoint

Model:

```text
models/sac_hardcore/ep_3000.pt
```

Evaluation Results:

| Metric              | Value  |
| ------------------- | ------ |
| Mean Return         | 300.22 |
| Standard Deviation  | 52.63  |
| Minimum Return      | -7.85  |
| Maximum Return      | 317.09 |
| Evaluation Episodes | 100    |

#### Observations

* The policy successfully exceeded the Hardcore solve threshold.
* Most evaluation runs achieved returns between 305 and 317.
* Several challenging seeds caused significant failures, increasing variance.
* Despite these failures, the average return remained above the solve threshold.

---

## Summary

| Environment              | Mean Return | Std Return | Min Return | Max Return | Solved |
| ------------------------ | ----------: | ---------: | ---------: | ---------: | :----: |
| BipedalWalker-v3         |      310.20 |      38.14 |      78.74 |     320.97 |   Yes  |
| BipedalWalkerHardcore-v3 |      300.22 |      52.63 |      -7.85 |     317.09 |   Yes  |

Both environments were successfully solved using the earliest checkpoint that exceeded the benchmark threshold. The repository includes the corresponding trained models and representative evaluation videos.
