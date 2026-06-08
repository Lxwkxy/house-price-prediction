# 🏡 House Price Prediction

A machine learning project focused on predicting real estate prices using advanced regression techniques and tree-based algorithms. This project demonstrates a complete Data Science workflow, from handling sparse categorical data to hyperparameter tuning for complex models.

## 🚀 Project Overview
The goal of this project is to accurately predict the `SalePrice` of houses based on various features such as living area, overall quality, and year built. The project highlights the transition from traditional linear models to advanced gradient boosting frameworks to capture non-linear relationships.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn

## 🧠 Methodology & Key Highlights

### 1. Data Preprocessing & Cleaning
* Handled missing values strategically (e.g., filling 'None' for properties without pools/garages, and median values for numerical columns).
* **Outlier Removal:** Identified and removed extreme outliers based on dataset documentation (e.g., properties with massive living areas but unusually low sale prices) to stabilize model training and prevent skewed predictions.
* Addressed the "Sparse Data Explosion" problem caused by categorical one-hot encoding.

### 2. Feature Engineering
Created highly impactful composite features that mirror real-world real estate valuation:
* `TotalSF`: Combined total square footage across all floors and basement.
* `TotalBath`: Aggregated full and half bathrooms.
* `HouseAge`: Calculated the true age of the house at the time of sale (`YrSold` - `YearBuilt`).

### 3. Feature Selection & Analysis
* Utilized **Mutual Information (MI)** to detect complex, non-linear relationships between features and the target variable, proving that standard correlation metrics are often insufficient for real estate data.

### 4. Modeling Strategy
* **Baseline Models:** Implemented `Ridge (L2)` and `Lasso (L1)` Regression to handle multicollinearity and perform automatic feature selection on sparse datasets.
* **Advanced Model:** Upgraded to **XGBoost (Extreme Gradient Boosting)** to capture non-linear interactions.
* **Hyperparameter Tuning:** Strictly tuned XGBoost parameters (`learning_rate`, `max_depth`, `subsample`) to prevent overfitting while maximizing predictive power on the validation set.

## 💻 How to Run

**Step 1: Clone the repository**

    git clone https://github.com/Lxwkxy/house-price-prediction.git

**Step 2: Install the required dependencies**

    pip install pandas numpy scikit-learn xgboost matplotlib seaborn

**Step 3: Run the pipeline**
Run the main notebook (`house_price.ipynb`) to preprocess the data, train the models, and generate the `submission.csv` file.

## 📌 Conclusion
This project serves as a comprehensive template for solving tabular data challenges, emphasizing the importance of domain-knowledge feature engineering, strategic data cleaning (outlier handling), and the transition from linear to tree-based algorithms.