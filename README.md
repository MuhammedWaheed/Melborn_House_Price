
# Melbourne House Price Prediction 🏡📈

This project implements a complete end-to-end machine learning pipeline to predict house prices in Melbourne.  
It includes data cleaning, feature engineering, preprocessing pipelines, model training, hyperparameter tuning,  
target transformation, and exporting the final CatBoost model.

---

## 📂 Project Files

- **Data_Cleaning.py** — Data cleaning, missing value handling, feature engineering  
- **Pipline.py** — ML preprocessing, pipelines, model training + evaluation  
- **model.pkl** — Final trained model  
- **cleaned_data.csv** — Cleaned dataset (generated from Data_Cleaning.py)  
- **requirements.txt** — All required packages  

---

## 🧹 Data Cleaning Highlights

- Removed irrelevant & high-missing columns  
- Extracted street names from addresses  
- Created new features (year, month, day, season)  
- Encoded categorical variables  
- Cleaned inconsistent values  
- Saved the final cleaned dataset  

---

## 🔧 Preprocessing Pipelines

The project uses **ColumnTransformer + Pipelines** to preprocess the data:

### Numerical
- SimpleImputer (most_frequent)
- StandardScaler

### Categorical
- TargetEncoder for high-cardinality features (suburb, sellerg, councilarea)
- OneHotEncoder for low-cardinality features (type, method, regionname, season)

---

## 🤖 Models Trained

- Linear Regression  
- KNN  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- **CatBoost (Best model!)**  

---

## 🎯 Target Transformation

Used to stabilize the price distribution:

```python
TransformedTargetRegressor(
    func=np.log1p,
    inverse_func=np.expm1
)
```

This greatly improves model performance and reduces skew.

---

## 🏆 Final Model Performance

After tuning and transformations:

- **Train R² ≈ 89%**  
- **Test R² ≈ 83%**  

This demonstrates strong generalization and low overfitting.

---

## 💾 Saving & Loading the Model

### Save:
```python
joblib.dump(model, 'model.pkl', compress=3)
```

### Load:
```python
model = joblib.load('model.pkl')
```

---

## 🚀 How to Run

1. Install dependencies  
```
pip install -r requirements.txt
```

2. Run cleaning script  
```
python Data_Cleaning.py
```

3. Run model training  
```
python Pipline.py
```

4. Use the model for predictions  

---

## 👨‍💻 Author  
Muhammed — Data Scientist & Machine Learning Engineer  
