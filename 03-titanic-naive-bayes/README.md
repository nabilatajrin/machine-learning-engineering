# Experiment 03 — Titanic Survival Prediction (Naive Bayes)

**Model:** Gaussian Naive Bayes  
**Features:** Pclass, Sex, Age, Fare  
**Target:** Survived (0/1)

---

## What this covers

1. Load Titanic data (local CSV or seaborn built-in)
2. Encode Sex (`female=0`, `male=1`)
3. Fill missing Age with mean
4. Train/test split (75/25)
5. `GaussianNB` fit + accuracy / RMSE / MAE / R²
6. Optional single-passenger prediction example

---

## Folder structure

```text
03-titanic-naive-bayes/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep          # place titanicsurvival.csv here if you have it
└── naive_bayes_titanic.py
```

If `data/titanicsurvival.csv` is missing, the script loads seaborn’s Titanic dataset automatically.

---

## Run

```bash
cd 03-titanic-naive-bayes
pip install -r requirements.txt
python naive_bayes_titanic.py
```

Expected accuracy is typically around **75–80%** depending on the split and data source.
