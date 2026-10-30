# House-Price-Prediction-
high-performance machine learning pipeline to predict house prices accurately using XGBoost and Scikit-Learn
# 🏡 Real Estate Price Prediction Pipeline

An end-to-end Machine Learning pipeline designed to predict real estate prices with high accuracy while strictly adhering to data science best practices (such as preventing data leakage and handling skewed features).

## 📊 Project Overview & Evolution
The project followed an iterative improvement approach, scaling up from a simple baseline model to an optimized gradient boosting framework:
1. **Linear Regression (Baseline):** Initial baseline to understand linear relationships.
2. **Random Forest Regressor:** Captured non-linear patterns, improving performance significantly.
3. **XGBoost Regressor:** Ensembled approach that pushed baseline accuracy further.
4. **Optuna-Tuned XGBoost (Final):** Hyperparameter-optimized model achieving a **Test R² Score of ~0.68** and **Train R² Score of ~0.76**, mitigating severe overfitting.

---

## 🛠️ Key Features & Engineering Steps
* **Custom Room Categorization:** Grouped raw room counts into meaningful ordinal categories (`Single / Couple`, `Small Family`, `Medium Family`, `Large Family`, `Joint Family`).
* **Outlier & Skewness Treatment:** Applied IQR-based clipping and log transformations (`np.log1p`) on right-skewed features like `Distance` and `Propertycount`.
* **Data Leakage Prevention:** Used Scikit-learn's **`ColumnTransformer`** to fit preprocessing steps exclusively on training data and transform test data safely.
* **Encoding Strategies:** 
  * `OrdinalEncoder` for ordinal features (`Room_Category`).
  * `OneHotEncoder` for nominal high-cardinality features (`Suburb`, `CouncilArea`, `Regionname`).
* **Feature Scaling:** Standardized numerical columns using `StandardScaler`.
* **Hyperparameter Optimization:** Leveraged **Optuna** (Bayesian Optimization) to systematically tune XGBoost parameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, etc.).

---

## 📈 Performance Summary
| Model | Train R² Score | Test R² Score |
| :--- | :---: | :---: |
| **Baseline (Linear Regression)** | - | ~0.51 - 0.61 |
| **Random Forest** | - | ~0.61 |
| **XGBoost (Default)** | - | ~0.64 |
| **Optuna-Tuned XGBoost** | **0.7605** | **0.6795** |

---

## 🧰 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Optuna

---

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
