
```markdown
# Assignment 2 – Language Model Training (Underfit, Overfit & Best Fit)

This project demonstrates three different training behaviors of a simple **Neural Language Model** built using **PyTorch**:

- **Underfitting**
- **Overfitting**
- **Best Fit (Optimal Training)**

The goal is to analyze how model complexity and training duration influence validation performance.

---

## 📁 Project Structure

```

Assignment2/
│── data/
│   └── input.txt
│
│── model.py
│── utils.py
│── train.py
│
│── plots/
│   ├── loss_underfit.png
│   ├── loss_overfit.png
│   ├── loss_bestfit.png
│
└── README.md

````

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install torch matplotlib numpy
````

### 2️⃣ Select the Scenario

Inside **train.py** set:

```python
EXPERIMENT = "underfit"
# or "overfit"
# or "bestfit"
```

### 3️⃣ Run Training

```bash
python train.py
```

---

# 📊 Output Plots

## ✅ Best Fit

![Best Fit](plots/loss_bestfit.png)

---

## ❌ Overfit

![Overfit](plots/loss_overfit.png)

---

## ⚠ Underfit

![Underfit](plots/loss_underfit.png)

---

# 🧠 Summary of Training Scenarios

| Scenario     | Train Loss | Validation Loss   | Interpretation                            |
| ------------ | ---------- | ----------------- | ----------------------------------------- |
| **Underfit** | High       | High              | Model is too small or trained too little  |
| **Overfit**  | Very low   | High (increasing) | Model memorizes training data             |
| **Best Fit** | Low        | Low               | Good balance of capacity & generalization |

---

# 🛠 Files Description

### `model.py`

Contains the language model architecture.

### `utils.py`

Handles preprocessing, batching, and dataset splitting.

### `train.py`

Main script that:

* Loads dataset
* Trains the model
* Generates loss plots

---

# ✨ Contact Information

**Name:** Your Name
**Mobile:** +91 9876543210
**Email:** [yourname@gmail.com](mailto:yourname@gmail.com)
**Google Drive Submission Link:** [https://drive.google.com/your-folder-link](https://drive.google.com/your-folder-link)



---

#
