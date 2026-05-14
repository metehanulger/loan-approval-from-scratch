[README_scratch.md](https://github.com/user-attachments/files/27763049/README_scratch.md)
# 🏦 Loan Approval Prediction — From Scratch

A Logistic Regression model built **without any ML libraries**, using only NumPy and pandas. Every component — train/test split, standard scaler, gradient descent, and evaluation metrics — is implemented from scratch.

---

## 📌 Overview

This project is a from-scratch implementation of the same loan approval prediction model from [loan-approval-prediction](https://github.com/metehanulger/loan-approval-prediction), but without using scikit-learn or any other ML library.

The goal is to understand what actually happens under the hood when you call `model.fit()` or `StandardScaler()`.

**Implemented from scratch:**
- `train_test_split`
- `StandardScaler` (fit, transform)
- `LogisticRegression` (sigmoid, gradient descent, predict)
- `accuracy_score`
- `classification_report` (precision, recall, f1-score)

---

## 📂 Project Structure

```
loan-approval-from-scratch/
│
├── main.py             # All implementation and training code
├── loan-train.csv      # Dataset (not included — see below)
└── README.md
```

---

## 📊 Dataset

The dataset used is the [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle.

Download it and place `loan-train.csv` in the project directory before running.

---

## ⚙️ How It Works

### 1. Data Preprocessing
Missing values filled with median (numeric) and mode (categorical). Categorical features one-hot encoded with `pd.get_dummies()`.

### 2. Train/Test Split
Shuffles row indices using `np.random`, splits 80/20 without any data leakage.

### 3. Standard Scaler
Computes mean and standard deviation from training data only, then normalizes:
```
z = (x - mean) / std
```

### 4. Logistic Regression
Uses sigmoid function and gradient descent to learn weights iteratively:
```
sigmoid(z) = 1 / (1 + e^(-z))
```
Weights updated each iteration to minimize prediction error.

### 5. Evaluation
Accuracy, precision, recall, and f1-score computed manually using true/false positives and negatives.

---

## 🚀 Getting Started

### Requirements

```bash
pip install pandas numpy
```

### Run

```bash
python main.py
```

### Output

```
--- Model Performance ---
Accuracy: %78.69

Detailed Report:
              precision     recall   f1-score    support
           0       0.95       0.42       0.58         43
           1       0.76       0.99       0.86         79
```

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy
- pandas

---

## 👤 Author

**Metehan Ülger**  
Software Engineering Student @ Ankara University  
[GitHub](https://github.com/metehanulger)
