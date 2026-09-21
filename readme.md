# DoorwayBench

A benchmark for predicting context-boundary forgetting from digital behavior.

## Overview

When people switch between apps, tabs, or tasks, they often forget what they were doing. This is the "digital doorway effect." DoorwayBench is the first benchmark for predicting this phenomenon from behavioral signals.

## Datasets

DoorwayBench unifies four public datasets under a common schema:

| Dataset | Events | Users | Domain | Label |
|---------|--------|-------|--------|-------|
| RLKWiC | 321 | 8 | Desktop | In-context (distraction) |
| Mindful Scrolling | 15,884 | 20 | Mobile | Mindful/Mindless |
| Kaggle Digital Behavior | 500 | — | Daily | Focus score |
| App Switch Networks | 53 | 53 | Graph | Structure |

## Task

Given a sequence of context-switch events, predict whether the user will forget their intended action:

**Input:** Sequence of events with temporal, contextual, behavioral, and graph features.
**Output:** P(forget at next switch).
**Label:** `forgetting_proxy` (0 = remembered, 1 = forgot).

## Results

| Model | AUC-ROC | AUC-PR | F1 |
|-------|---------|--------|-----|
| XGBoost | 0.700 | 0.466 | 0.402 |
| Random Forest | 0.697 | 0.521 | 0.371 |
| TCN | 0.675 | 0.462 | 0.397 |
| LSTM | 0.655 | 0.427 | 0.428 |
| Transformer | 0.650 | 0.445 | 0.390 |
| Logistic Regression | 0.634 | 0.411 | 0.371 |

**Key finding:** Tree-based models outperform neural sequence models at the current data scale (n=321). No pairwise model differences are statistically significant. Power analysis shows the benchmark can only detect AUC differences ≥ 0.192 at 80% power.

## Setup

```bash
pip install -r requirements.txt