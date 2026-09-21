# DoorwayBench: One-Page Summary

## Problem
People forget what they were doing after switching apps or tabs. This "digital doorway effect" is universal but has no benchmark.

## What We Built
- Unified 4 public datasets under one schema
- Engineered 23 features (temporal, contextual, behavioral, graph)
- Trained 7 models (LR, RF, XGBoost, LSTM, TCN, Transformer, Contrastive)
- Evaluated with 5-fold CV, bootstrap CIs, and power analysis

## Key Findings
1. XGBoost achieves the best AUC-ROC (0.700)
2. No pairwise model differences are statistically significant
3. Sequence models underperform feature-based models at n=321
4. Contrastive pre-training provides marginal improvement (+0.03)
5. Benchmark is underpowered: needs 5,510 events to detect ΔAUC=0.05

## Contribution
First benchmark for context-boundary forgetting. Open dataset, code, and evaluation protocol.

## What's Next
- Collect 5,510+ events from 50+ users
- Add mobile datasets (AWARE)
- Build real-time prototype
- Try supervised contrastive learning

## Links
- Code: https://github.com/YOUR_USERNAME/doorway-bench
- Data: https://github.com/YOUR_USERNAME/doorway-bench/tree/main/data