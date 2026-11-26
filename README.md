# Machine Learning Engineering

> End-to-end Machine Learning projects focusing on production-ready practices, MLOps, and scalable systems.

**Author:** [Nabila Tajrin](https://github.com/nabilatajrin)

## Overview
This repository showcases practical ML Engineering skills — from data preprocessing to model deployment, monitoring, and scaling.

## Experiments & labs

| Folder | Description |
|--------|-------------|
| [`01-linear-regression`](01-linear-regression/) | **Experiment 01** — manual + sklearn linear regression |
| [`02-decision-trees-ensemble`](02-decision-trees-ensemble/) | **Experiment 02** — decision trees & ensembles (Iris) |
| [`03-titanic-naive-bayes`](03-titanic-naive-bayes/) | **Experiment 03** — Titanic survival with GaussianNB |

```bash
# Experiment 01
cd 01-linear-regression && pip install -r requirements.txt && python linear_regression.py

# Experiment 02
cd ../02-decision-trees-ensemble && pip install -r requirements.txt
python 01_decision_tree_iris.py && python 02_ensemble_methods_iris.py

# Experiment 03
cd ../03-titanic-naive-bayes && pip install -r requirements.txt && python naive_bayes_titanic.py
```

## Other topics
- Feature engineering and data pipelines
- Supervised learning with scikit-learn
- Exploratory data analysis
- MLOps fundamentals
- End-to-end projects

## Tech Stack
- Python
- Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- MLflow, Docker, FastAPI (MLOps paths)

## Installation
```bash
git clone https://github.com/nabilatajrin/machine-learning-engineering.git
cd machine-learning-engineering
```

## License
MIT © Nabila Tajrin
