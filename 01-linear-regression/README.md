# Experiment 01 — Linear Regression

**Author:** Nabila Tajrin Bristy  
**Topic:** Simple linear regression from scratch and with scikit-learn

---

## What this covers

1. **Manual formulas** — slope \(m\) and intercept \(c\) using sums  
2. **scikit-learn** — `LinearRegression`, train/test split  
3. **Metrics** — R², MSE, MAE  
4. **Plot** — training points, test points, regression line  

Data (from the lab notebook):

```text
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]
```

---

## Folder structure

```text
01-linear-regression/
├── README.md
├── requirements.txt
└── linear_regression.py
```

---

## Run

```bash
cd 01-linear-regression
pip install -r requirements.txt
python linear_regression.py
```

A plot window opens (or is saved if you are on a headless environment).

---

## Formulas

$$
m = \frac{n\sum xy - (\sum x)(\sum y)}{n\sum x^2 - (\sum x)^2}
$$

$$
c = \frac{\sum y - m\sum x}{n}
$$
