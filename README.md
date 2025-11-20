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
├── Data_Cleaning.py           # Data cleaning & feature engineering
├── Pipline.py                 # Preprocessing pipelines & model training
├── model.pkl                  # Final trained model (CatBoost + target transform)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

> 💡 You can adjust folder names/paths to match your own repository layout.

---

## 📊 Exploratory Data Analysis (EDA)

In the EDA phase, we explore the target (`Price`) and key predictors.

### 🔹 Price Distribution

A histogram of house prices shows a **right-skewed distribution** with a long tail of expensive properties:

```python
plt.figure(figsize=(10, 5))
plt.hist(df['Price'], bins=50)
plt.title('Distribution of Property Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```

**Example visualization (replace with your own image file):**

![Price Distribution](images/price_distribution.png)

---

### 🔹 Landsize Distribution

Land size also exhibits strong skewness with a few very large properties:

```python
plt.figure(figsize=(10, 5))
plt.hist(df['Landsize'], bins=50)
plt.title('Distribution of Landsize')
plt.xlabel('Landsize (m²)')
plt.ylabel('Frequency')
plt.show()
```

Visualization:

![Landsize Distribution](images/landsize_distribution.png)

---

### 🔹 Rooms vs Price

The relationship between number of rooms and price is explored using boxplots and scatter plots:

```python
sns.boxplot(x='Rooms', y='Price', data=df)
plt.title('Price Distribution by Number of Rooms')
plt.show()
```

Visualization:

![Rooms vs Price](images/rooms_vs_price.png)

---

### 🔹 Distance vs Price

We also inspect how distance from the CBD affects price:

```python
plt.figure(figsize=(10, 5))
plt.scatter(df['Distance'], df['Price'], alpha=0.3)
plt.title('Price vs Distance from CBD')
plt.xlabel('Distance (km)')
plt.ylabel('Price')
plt.show()
```

Visualization:

![Distance vs Price](images/distance_vs_price.png)

> 📝 Replace the `images/*.png` paths with the filenames of your own saved plots in your repository (e.g., `images/price_dist.png`, etc.).

---

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

---

## 🔧 Preprocessing Pipelines

Preprocessing is implemented using `Pipeline` and `ColumnTransformer` from `scikit-learn` to ensure consistent and reproducible transformations.

### 🔹 Numerical Features

Two numerical pipelines:

```python
num_pipeline1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

num_pipeline2 = Pipeline(steps=[
    ('scaler', StandardScaler())
])
```

### 🔹 Categorical Features

High-cardinality categorical features (e.g. `Suburb`, `SellerG`, `CouncilArea`) are handled using **Target Encoding**, while low-cardinality ones use **One-Hot Encoding**:

```python
cat_pipeline1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_encoder', TargetEncoder())
])

cat_pipeline2 = Pipeline(steps=[
    ('target_encoder', TargetEncoder())
])

cat_pipeline3 = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])
```

### 🔹 ColumnTransformer

All pipelines are combined into a single preprocessing object:

```python
preprocessing = ColumnTransformer(
    transformers=[
        ('num_pipeline1', num_pipeline1, ['Car', 'YearBuilt']),
        ('num_pipeline2', num_pipeline2, ['Rooms', 'Distance', 'Bedroom2',
                                          'Bathroom', 'Landsize', 'year', 'month', 'day']),
        ('cat_pipeline1', cat_pipeline1, ['CouncilArea']),
        ('cat_pipeline2', cat_pipeline2, ['Suburb', 'SellerG']),
        ('cat_pipeline3', cat_pipeline3, ['Type', 'Method', 'Regionname', 'season'])
    ],
    remainder='drop'
)
```

---

## 🤖 Models & Training

Several models were evaluated using a unified pipeline:

- Linear Regression  
- KNN  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- **CatBoost (final chosen model)**  

Example setup:

```python
models = [
    ('Linear Regression', LinearRegression(n_jobs=-1)),
    ('KNN', KNeighborsRegressor()),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42, n_jobs=-1)),
    ('XGBoost', XGBRegressor()),
    ('CatBoost', CatBoostRegressor(verbose=0)),
    ('LightGBM', LGBMRegressor())
]

for name, reg in models:
    model_pipeline = Pipeline(steps=[
        ('Preprocessing', preprocessing),
        ('Model', reg)
    ])
    ...
```

---

## 🎯 Target Transformation

House prices are **highly skewed**, so a `TransformedTargetRegressor` is used:

```python
from sklearn.compose import TransformedTargetRegressor
import numpy as np

model_pipeline = Pipeline(steps=[
    ('Preprocessing', preprocessing),
    ('Model', CatBoostRegressor(learning_rate=0.1, depth=6, l2_leaf_reg=9, verbose=0))
])

final_model = TransformedTargetRegressor(
    regressor=model_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)
```

This helps:

- Stabilize variance  
- Reduce the impact of very expensive properties  
- Improve model performance and generalization  

---

## 🔍 Hyperparameter Tuning (CatBoost)

Hyperparameters tuned with `RandomizedSearchCV`:

```python
param_grid = {
    'regressor__Model__depth': [4, 6, 8, 10],
    'regressor__Model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__Model__l2_leaf_reg': [1, 3, 5, 7, 9]
}

random_search = RandomizedSearchCV(
    final_model,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    return_train_score=True
)
```

---

## 🏆 Final Model & Performance

The final chosen model is:

- **CatBoostRegressor with tuned hyperparameters**
- Wrapped in:
  - `Pipeline` (for preprocessing)
  - `TransformedTargetRegressor` (for log-scaling the target)

Typical performance:

- **Train R² ≈ 0.89 (89%)**  
- **Test R² ≈ 0.83 (83%)**

This indicates **strong generalization** without severe overfitting.

---

## 💾 Saving & Loading the Model

```python
import joblib

# Save
joblib.dump(final_model, 'model.pkl', compress=3)

# Load
model = joblib.load('model.pkl')

# Predict
model.predict(X.head(1))
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Data Cleaning

```bash
python Data_Cleaning.py
```

This will generate `cleaned_data.csv`.

### 3️⃣ Train the Model

```bash
python Pipline.py
```

This will train the model and save `model.pkl`.

### 4️⃣ Use the Model for Inference

```python
import joblib
import pandas as pd

model = joblib.load('model.pkl')
new_data = pd.read_csv('some_new_data.csv')  # structured like training features
preds = model.predict(new_data)
```

---

## 📝 Notes on Visualizations

This README references images under an `images/` folder:

- `images/price_distribution.png`
- `images/landsize_distribution.png`
- `images/rooms_vs_price.png`
- `images/distance_vs_price.png`

👉 **Action for you:**  
Place your plot images in an `images/` folder in your repo, and either:
- Rename them to match these filenames, or  
- Update the image paths in this README accordingly.

---

## 👨‍💻 Author

**Muhammed**  
_Data Scientist & Machine Learning Enthusiast_

Feel free to fork, open issues, or suggest improvements!
