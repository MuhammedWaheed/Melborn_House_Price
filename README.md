# 🤖 Test App From this link 
(https://melbornhouseprice-khjawg7csmnhxqvpqmssrh.streamlit.app/)


# 🏡 Melbourne House Price Prediction

An end-to-end machine learning project to predict **house prices in Melbourne, Australia** using:
- Cleaned + engineered tabular data  
- Robust preprocessing pipelines  
- Advanced gradient boosting models (CatBoost, XGBoost, LightGBM)  
- Target transformation (log1p / expm1)  
- Hyperparameter tuning  
- A fully exportable production-ready model (`model.pkl`)

---

## 📌 Project Overview

The goal of this project is to build a **reliable regression model** that can predict the sale price of houses in Melbourne based on features such as:
- Location (suburb, region, council area, distance from CBD)  
- Property characteristics (rooms, bathrooms, car spots, landsize, year built)  
- Sale information (method of sale, season, seller agency)  

The project is structured to reflect **good real-world ML practices**:
1. Data cleaning & feature engineering  
2. Exploratory data analysis (EDA) with visualizations  
3. Preprocessing using `Pipeline` + `ColumnTransformer`  
4. Model comparison across several algorithms  
5. Use of `TransformedTargetRegressor` for skewed targets  
6. Hyperparameter tuning for the best model  
7. Model persistence using `joblib`  

---

## 📂 Project Structure

```bash
.
├── data/
│   ├── melb_data.csv          # Raw dataset
│   └── cleaned_data.csv       # Clean dataset after preprocessing
│
├── Data_Cleaning.ipynb          # Data cleaning & feature engineering
├── Pipline.ipynb                 # Preprocessing pipelines & model training
├── model.pkl                  # Final trained model (CatBoost + target transform)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---
.







## 🧹 Data Cleaning & Feature Engineering

All cleaning is handled in **`Data_Cleaning.py`**, including:

- Handling missing values  
- Dropping high-missing or low-information columns  
- Parsing the `Date` feature into:
  - `year`, `month`, `day`, `season`
- Cleaning and splitting the `Address` into:
  - `street_name` (later dropped for modeling to avoid high cardinality noise)  
- Ensuring correct data types for numeric and categorical variables  
- Exporting the cleaned dataset to `cleaned_data.csv`

Example of season extraction:

```python
df['season'] = df['month'] % 12 // 3 + 1
```


## 🤖 Models & Training

Several models were evaluated using a unified pipeline:

- Linear Regression  
- KNN  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- **CatBoost (final chosen model)**  



## 🎯 Target Transformation

House prices are **highly skewed**, so a `TransformedTargetRegressor` is used:

- Stabilize variance  
- Reduce the impact of very expensive properties  
- Improve model performance and generalization  

---

## 🔍 Hyperparameter Tuning (CatBoost)

Hyperparameters tuned with `RandomizedSearchCV`:


## 🏆 Final Model & Performance

The final chosen model is:

- **CatBoostRegressor with tuned hyperparameters**
- Wrapped in:
  - `Pipeline` (for preprocessing)
  - `TransformedTargetRegressor` (for log-scaling the target)

**Muhammed**  
_Data Scientist & Machine Learning Enthusiast_
