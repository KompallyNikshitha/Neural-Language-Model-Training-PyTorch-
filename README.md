
```markdown
# Assignment 2 – Language Model Training (Underfit, Overfit & Best Fit)

This project demonstrates three different training behaviors of a simple **Neural Language Model** built using **PyTorch**:

- **Underfitting**
- **Overfitting**
- **Best Fit (Optimal Training)**

The goal is to analyze how model complexity and training duration influence validation performance.

---
```
## 🚀 How to Run

## **1️⃣ Install Dependencies**

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
### 4️⃣ Generate Report
```bash
python report.py
```

---

# 📊 Output Plots

## ✅ Best Fit
<img width="640" height="480" alt="loss_bestfit" src="https://github.com/user-attachments/assets/747fbc84-073e-4397-90be-f83311bd7c1b" />

---

## ❌ Overfit
<img width="640" height="480" alt="loss_overfit" src="https://github.com/user-attachments/assets/64ba9771-2030-49c7-84ab-7c0989d11e1f" />

---

## ⚠ Underfit
<img width="640" height="480" alt="loss_underfit" src="https://github.com/user-attachments/assets/19e7fdaf-7601-4720-9777-46525537ece2" />

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

**Name:** Nikshitha Kompally

**Mobile:** +91 9701495508

**Email:** [nikshithakompally08@gmail.com](mailto:nikshithakompally08@gmail.com)

**Google Drive Submission Link:** [https://drive.google.com/drive/folders/1J_G7dBAwUTf5eAOwzpHHO1ogRTXArjMa?usp=drive_link](https://drive.google.com/drive/folders/1J_G7dBAwUTf5eAOwzpHHO1ogRTXArjMa?usp=drive_link)



---

#
