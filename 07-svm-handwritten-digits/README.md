# Experiment 07 — SVM Handwritten Digit Recognition

**Dataset:** scikit-learn `load_digits()` (8×8 handwritten digits, 1797 samples)  
**Models:** SVC with linear / RBF kernels and different `gamma` / `C`  
**Author:** Nabila Tajrin Bristy

---

## What this covers

1. Load digits dataset and inspect shapes
2. Visualize a sample digit image
3. Flatten images → feature matrix `(n_samples, 64)`
4. Train/test split (75/25)
5. Train four SVM variants and compare accuracy
6. Predict a single sample digit

---

## Folder structure

```text
07-svm-handwritten-digits/
├── README.md
├── requirements.txt
└── svm_digits.py
```

---

## Run

```bash
cd 07-svm-handwritten-digits
pip install -r requirements.txt
python svm_digits.py
```

Typical accuracies (approx.):

| Model | Kernel / params | Accuracy |
|-------|-----------------|----------|
| Model 1 | `kernel='linear'` | ~98.2% |
| Model 2 | `kernel='rbf'` | ~98.7% |
| Model 3 | `gamma=0.001` | ~98.9% |
| Model 4 | `gamma=0.001, C=0.1` | ~96.4% |
