# Binary Classification Pipeline Search: 29 Combinations Compared by AUC

A systematic search over feature engineering, imputation, resampling, feature selection, and model/ensemble choices for a binary classification problem (predicting the probability of a financial default), scored by AUC on a held-out test set.

## The problem

A financial dataset with an anonymized feature set (`X2`...`X78`) and a binary target `Y` (default vs. non-default). The features carry no semantic labels by design, so this is a pure signal-extraction exercise: no domain shortcuts, just systematic experimentation across the ML pipeline.

## Approach

Rather than hand-tuning one model, I ran 29 distinct pipeline combinations varying:

- **Imputation**: Simple (mean/median/mode), KNN, Iterative
- **Scaling**: Standard, MinMax, Robust
- **Feature reduction/selection**: PCA, RFE, variance threshold, correlation-based selection, mutual information, permutation importance, and model-based selection (using XGBoost/LightGBM/Random Forest feature importances as a filter)
- **Resampling**: SMOTE, ADASYN, Tomek Links (for class imbalance)
- **Models**: XGBoost, LightGBM, CatBoost, Random Forest, AdaBoost, both standalone and combined via Stacking and (weighted) soft Voting
- **Hyperparameter tuning**: Optuna (TPE sampler)

Every combination's predictions were exported separately (see [`predictions/`](predictions/)) so results are directly comparable and reproducible.

## Results

### XGBoost, standalone and with feature selection

| Combination | Pipeline | AUC |
|---|---|---|
| 1 | Standard scaling + XGBoost (default) | 0.9395 |
| 2 | PCA + MinMax + SimpleImputer + XGBoost | 0.8685 |
| 4 | Median imputer + PCA + Standard scaling + XGBoost | 0.8772 |
| 5 | KNN imputer + Standard scaling + XGBoost (Optuna) | 0.9510 |
| 6 | KNN imputer + SMOTE + Variance threshold + XGBoost (Optuna) | 0.8948 |
| 7 | KNN imputer + PCA + Standard scaling + XGBoost (Optuna) | 0.9103 |
| 8 | KNN imputer + SMOTE + XGBoost (Optuna) | 0.8969 |
| 10 | KNN imputer + Standard scaling + XGBoost (Optuna, TPE) | 0.9500 |
| 13 | XGBoost (vanilla) + permutation feature importance | 0.9349 |
| 17 | XGBoost (Optuna) + permutation feature importance | 0.9446 |
| 18 | LightGBM feature selection + XGBoost (Optuna) | 0.9549 |
| 19 | Random Forest feature selection + XGBoost (Optuna) | 0.9404 |
| **23** | **XGBoost feature selection + XGBoost (Optuna)** | **0.9550** (best standalone) |

### Stacking ensembles

| Combination | Pipeline | AUC |
|---|---|---|
| 14 | Stacking: XGBoost + LightGBM + CatBoost (untuned) | 0.9269 |
| 15 | Stacking: XGBoost + LightGBM + CatBoost (Optuna) | 0.9543 (best stacking) |

### Voting ensembles

| Combination | Pipeline | AUC |
|---|---|---|
| 9 | KNN imputer + Standard scaling + Voting (LGBM, RF, XGB, Optuna) | 0.9476 |
| 16 | Voting: XGBoost + LightGBM + AdaBoost (weighted, untuned) | 0.9533 |
| **22** | **Voting: LightGBM + XGBoost + AdaBoost (weighted 2:1.5:1), LightGBM feature selection, LGBM+XGB Optuna-tuned** | **0.9569 (best overall)** |
| 29 | Voting: XGBoost + LightGBM + AdaBoost, XGBoost feature selection | 0.9568 |

**Best overall: Combination 22**, implemented in [`best_model.py`](best_model.py).

## Key findings

- **XGBoost was the strongest single learner** across almost every variant tried.
- **Optuna tuning consistently helped**; the untuned baselines (Combinations 2, 4) were the weakest results in the whole search.
- **Model-based feature selection beat classical dimensionality reduction.** Using LightGBM or XGBoost feature importances as a selection filter outperformed PCA, variance threshold, and RFE by a wide margin (compare Combination 18/23 vs. 2/4/7).
- **Scalers, imputers, and resampling mostly hurt performance** when layered on for their own sake — KNN imputation was the one exception that consistently helped, and was carried into every later combination.
- **SMOTE/ADASYN did not help** on this dataset, despite being an intuitive choice for a classification problem; both variants underperformed simpler pipelines.
- **Ensembling paid off**, but with a time cost: stacking in particular was slow enough that a planned AdaBoost variant never got tuned. The weighted voting ensemble in Combination 22 edged out the best single model and the best stacking model, and was cheaper to iterate on than stacking.

## What I'd do with more time

- Tune AdaBoost's hyperparameters (it was used with reasonable defaults, untuned, in every ensemble) and add it to the stacking ensemble, not just voting.
- Sweep the LightGBM feature-selection importance threshold more systematically — Combination 22 used a threshold of 0, and the writeup for Combination 18 suggests results improved as the threshold was tuned, but this wasn't explored exhaustively.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── best_model.py                  # Combination 22, the best-performing pipeline
├── notebooks/
│   └── pipeline_experiments.ipynb # all 29 combinations, end to end
└── predictions/                   # per-combination prediction outputs (ID + predicted probability)
```

**Note on data:** the raw training/test datasets are course-provided and not included here. `best_model.py` and the notebook expect `fda_trainingset.csv` (features `X2`...`X78`, `ID`, target `Y`) and `fda_testset.csv` (same features + `ID`, no target) in the working directory.

## Stack

Python · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · imbalanced-learn
