# House Rent Prediction

![Project Workflow](https://img.shields.io/badge/Pipeline-EDA%20%7C%20FE%20%7C%20Modeling%20%7C%20Evaluation-brightgreen)

## 🚀 Project Overview
This repository contains an end-to-end machine learning project for predicting monthly rents of properties based on features such as square footage, number of bedrooms/bathrooms, region, and various amenities. The entire workflow is implemented in a Jupyter Notebook, covering:

- Data acquisition via Kaggle API  
- Exploratory Data Analysis (EDA)  
- Feature Engineering (FE)  
- Model training & hyperparameter tuning (Linear Regression, Random Forest, XGBoost)  
- Model evaluation (RMSE, MAE, residual analysis)  

## 📊 Dataset
The dataset is sourced from Kaggle: [House Rent Prediction Dataset](https://www.kaggle.com/datasets/rkb0023/houserentpredictiondataset). It includes 265,190 listings from Craigslist across different US regions.

| Column                  | Description                                    |
|-------------------------|------------------------------------------------|
| `price`                 | Monthly rent in USD                            |
| `sqfeet`                | Square footage of the property                 |
| `beds`                  | Number of bedrooms                             |
| `baths`                 | Number of bathrooms                            |
| `cats_allowed`          | Cats allowed (0/1)                             |
| `dogs_allowed`          | Dogs allowed (0/1)                             |
| `laundry_options`       | Laundry availability (e.g., in-unit, on-site)  |
| `parking_options`       | Parking type (e.g., street, garage)            |
| `region`                | Geographic region (city or metro area)         |
| `type`                  | Property type (apartment, house, etc.)         |

Other columns (URLs, descriptions, images, etc.) are dropped during feature engineering.


## 📝 Results

Add the following Python snippet to compute and display your final evaluation metrics in the README:

```python
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# assume lr, gs_rf, gs_xg, X_test, y_test are already defined
models = {
    'Linear Regression': lr,
    'Random Forest'    : gs_rf.best_estimator_,
    'XGBoost'          : gs_xg.best_estimator_
}

# compute metrics
rows = []
for name, model in models.items():
    preds = model.predict(X_test)
    mse  = mean_squared_error(y_test, preds)
    rmse = mse**0.5
    mae  = mean_absolute_error(y_test, preds)
    rows.append({
        'Model' : name,
        'RMSE'  : f"{rmse:,.0f}",
        'MAE'   : f"{mae:,.0f}"
    })

# display as table
results_df = pd.DataFrame(rows)
print(results_df.to_markdown(index=False))

| Model              | RMSE    | MAE   |
|--------------------|---------|-------|
| Linear Regression  | 39,297  | 1,770 |
| Random Forest      | 21,422  | 156   |
| XGBoost            | 74,343  | 899   |

