```markdown
# Assignment 2 – Language Model Training (Underfit, Overfit & Best Fit)

This project implements and compares three training behaviors of a simple **Neural Language Model** using **PyTorch**:

- **Underfitting**
- **Overfitting**
- **Best Fit (Optimal Generalization)**

The objective is to understand how training configuration, dataset size, and model capacity affect generalization.

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

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install torch matplotlib numpy
````

### 2️⃣ Choose the Training Scenario

Open **train.py** and set:

```python
EXPERIMENT = "underfit"
# or
EXPERIMENT = "overfit"
# or
EXPERIMENT = "bestfit"
```

### 3️⃣ Run the Training

```bash
python train.py
```

Each run will automatically save the corresponding plot in the **plots/** folder.

---

# 📊 Training Result Plots

## ✅ Best Fit

Training loss and validation loss decrease steadily → good generalization.

![Best Fit](plots/loss_bestfit.png)

---

## ❌ Overfit

Training loss keeps decreasing, but validation loss increases → model memorizes data.

![Overfit](plots/loss_overfit.png)

---

## ⚠ Underfit

Both training and validation losses remain high → model too simple or trained too little.

![Underfit](plots/loss_underfit.png)

---

# 🧠 Summary of the Three Scenarios

| Scenario     | Train Loss      | Validation Loss            | Explanation                                           |
| ------------ | --------------- | -------------------------- | ----------------------------------------------------- |
| **Underfit** | Slight decrease | Stagnant / slightly rising | Model is too simple or training is too short          |
| **Overfit**  | Very low        | High and rising            | Model memorizes training data but fails to generalize |
| **Best Fit** | Smooth decrease | Smooth decrease            | Balanced capacity → best performance                  |

---

# 🛠 Code Overview

## `model.py`

Defines the neural network architecture:

* Embedding layer
* Hidden linear layer
* ReLU activation
* Output projection

## `utils.py`

Handles:

* Text preprocessing
* Dataset batching
* Train/validation split

## `train.py`

Responsible for:

* Loading data
* Selecting experiment type
* Running training loop
* Saving loss plots

---

# 🎯 Learning Outcomes

By completing this assignment, you learn:

* What causes **underfitting** and **overfitting**
* How to control model capacity and epochs
* How to evaluate model performance using **loss curves**
* How training settings impact generalization

---

# ✨ Author

Nikshitha Kompally

---



```
