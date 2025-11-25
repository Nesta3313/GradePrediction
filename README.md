#  **Student Grade Prediction — Regression Modeling (Linear Regression + ElasticNet)**

## Overview

This project builds machine learning models to **predict a student’s final grade (G3)** based on demographic, behavioral, and academic performance features from the *Student Performance* dataset.

The workflow demonstrates a complete, production-style machine learning pipeline:

* Data preprocessing
* Train/validation/test split
* Feature scaling
* Model training (Linear Regression, ElasticNet)
* Hyperparameter tuning (GridSearchCV)
* Evaluation using multiple metrics
* Diagnostic analysis (residual plots & distributions)
* Coefficient interpretation
* Model comparison and selection

This project emphasizes **interpretability**, **robustness**, and **proper evaluation practices**.

---

## Dataset

Source: https://archive.ics.uci.edu/dataset/320/student+performance

**Features Used**

* `age`
* `sex` (encoded: F=0, M=1)
* `studytime`
* `absences`
* `G1` – first period grade
* `G2` – second period grade
* **G3** – *final grade (target)*

Shape: **395 rows × 7 columns**

---

# **Methods & Tools**

### Libraries

* pandas, numpy
* matplotlib, seaborn
* scikit-learn (Pipeline, ColumnTransformer, LinearRegression, ElasticNet, GridSearchCV)

---

## **Modeling Approach**

### Train/Validation/Test Split (70/15/15)

We correctly split the data twice:

1. Train vs. Temp
2. Temp → Validation + Test

This avoids leaking validation information into training.

### Preprocessing Pipeline

A `ColumnTransformer` + `Pipeline` structure ensures:

* Numeric features are standardized
* Categorical features remain consistent
* No data leakage
* Clean deployment-ready modeling

---

# **Baseline Model: Linear Regression**

### **Test Performance**

| Metric   | Score |
| -------- | ----- |
| **RMSE** | 1.85  |
| **MAE**  | 1.22  |
| **MSE**  | 3.42  |
| **R²**   | 0.83  |

### Interpretation

* Very strong performance for a simple linear model.
* The model slightly over-relies on **G2**, the second-period exam grade, which has strong correlation with G3.
* Residual plots and distribution confirm healthy model behavior.

---

# **Regularized Model: ElasticNet (Chosen Model)**

After tuning via **GridSearchCV**, the best hyperparameters were:

```
alpha = 0.1
l1_ratio = 0.9
```

This corresponds to a model that is **mostly Lasso**, with slight Ridge smoothing.

---

## **ElasticNet — Test Performance (Final Chosen Model)**

| Metric   | Score    |
| -------- | -------- |
| **RMSE** | **1.79** |
| **MAE**  | **1.13** |
| **MSE**  | 3.21     |
| **R²**   | **0.84** |

### ElasticNet OUTPERFORMED Linear Regression

* Better predictive accuracy
* Better generalization
* Shrinks noisy features
* Handles multicollinearity between `G1` and `G2`
* Produces a more stable, robust model

---

# **Diagnostic Plots**

### Actual vs. Predicted (ElasticNet)

* Points closely follow the perfect-prediction diagonal
* Indicates good accuracy across the grade range

### Residual Scatter Plot

* Residuals randomly distributed around zero
* No heteroscedasticity
* No curvature
* A few extreme outliers (normal for student datasets)

###️ Residual Histogram

* Roughly bell-shaped distribution
* Confirms valid linear modeling assumptions
* No need for target transformation (despite skewed G3)

---

# **Coefficient Interpretation**

| Feature       | Linear Regression | ElasticNet (Final) | Interpretation                                    |
| ------------- | ----------------- | ------------------ |---------------------------------------------------|
| **age**       | -0.265            | -0.169             | Older students perform slightly worse             |
| **sex**       | +0.066            | **0.000**          | ElasticNet determined sex has no predictive value |
| **studytime** | -0.018            | **0.000**          | Probably self-reported, noisy → removed by L1     |
| **absences**  | +0.329            | +0.215             | Still mildly predictive                           |
| **G1**        | +0.606            | +0.600             | First exam is a strong predictor                  |
| **G2**        | +3.590            | +3.480             | Second exam is the strongest predictor            |

### ElasticNet shrank:

* noisy features to **zero**
* inflated features (G1, G2) slightly
* this makes the model **more reliable and less sensitive**

---

# **Final Model Choice**

### **ElasticNet is chosen as the final model**

Because it provides:

* Higher predictive accuracy
* Better bias–variance tradeoff
* More robust coefficients
* Feature selection via coefficient shrinkage
* Better generalization to unseen data

---

#  License

This project is for educational and portfolio demonstration purposes.

