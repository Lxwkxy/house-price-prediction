# 🏡 House Price Prediction

A machine learning project focused on predicting real estate prices using advanced regression techniques and tree-based algorithms. This project demonstrates a complete Data Science workflow, from handling sparse categorical data to implementing a robust Stacking Ensemble model.

## 🚀 Project Overview
The goal of this project is to accurately predict the `SalePrice` of houses based on various features such as living area, overall quality, and year built. The project highlights the transition from traditional linear models to an advanced ensemble framework to capture both linear trends and non-linear relationships.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn

## 🧠 Methodology & Key Highlights

### 1. Data Preprocessing & Cleaning
* Handled missing values strategically (e.g., filling 'None' for properties without pools/garages, and median values for numerical columns).
* **Outlier Removal:** Identified and removed extreme outliers based on dataset documentation (e.g., properties with massive living areas but unusually low sale prices) to stabilize model training and prevent skewed predictions.
* Addressed the sparse data explosion problem caused by categorical one-hot encoding by carefully aligning train and test datasets.

### 2. Feature Engineering
Created highly impactful composite features that mirror real-world real estate valuation:
* `TotalSF`: Combined total square footage across all floors and basement.
* `TotalBath`: Aggregated full and half bathrooms.
* `HouseAge`: Calculated the true age of the house at the time of sale (`YrSold` - `YearBuilt`).

### 3. Feature Selection & Analysis
* Utilized **Mutual Information (MI)** to detect complex, non-linear relationships between features and the target variable, proving that standard correlation metrics are often insufficient for real estate data.

### 4. Advanced Modeling Strategy (Stacking Ensemble)
Implemented a sophisticated Stacking Regressor to maximize predictive accuracy:
* **Tree-Based Base Model:** `XGBRegressor` tuned with strict hyperparameters (`learning_rate`, `max_depth`, `subsample`) to capture complex, non-linear interactions without overfitting.
* **Linear Base Models via Pipeline:** `RidgeCV` (L2) and `LassoCV` (L1) regressions integrated with `StandardScaler` inside a scikit-learn `Pipeline`. This ensures scale-sensitive algorithms receive properly standardized data seamlessly during cross-validation, while keeping the global variables untouched.
* **Meta-Model:** A final `RidgeCV` estimator combines the predictions of the base models, effectively learning when to trust the tree-based model versus the linear models.

## 💻 How to Run

**Step 1: Clone the repository**

    git clone https://github.com/Lxwkxy/house-price-prediction.git

**Step 2: Install the required dependencies**

    pip install pandas numpy scikit-learn xgboost matplotlib seaborn

**Step 3: Run the pipeline**
Run the main notebook (`house_price.ipynb`) to preprocess the data, train the ensemble model, and generate the `submission_ensemble_fixed.csv` file.

## 📌 Conclusion
This project serves as a comprehensive template for solving tabular data challenges. It emphasizes the importance of domain-knowledge feature engineering, strategic data cleaning (outlier handling), and the architectural design of a robust Machine Learning pipeline using ensemble techniques.