# house-price-prediction
End-to-end house price prediction using Random Forest, XGBoost, LightGBM, hyperparameter tuning, model stacking, and SHAP explainability.




# 🏠 House Price Prediction

An end-to-end machine learning project for predicting house sale prices using advanced feature engineering, ensemble models, hyperparameter tuning, and model explainability.

## 📌 Project Objective

The goal of this project is to predict the sale price of residential houses using the Kaggle House Prices dataset.

The project focuses not only on model accuracy, but also on:
- Data preprocessing
- Feature engineering
- Model comparison
- Hyperparameter optimization
- Ensemble learning
- Explainability using SHAP
- Saving the final model for future predictions

## 📊 Dataset

Dataset: Kaggle House Prices – Advanced Regression Techniques

Training data:
- 1,460 houses
- 80 original features
- Target: `SalePrice`

The dataset contains numerical and categorical information about houses, including:
- Overall quality
- Living area
- Basement area
- Garage information
- Year built
- Neighborhood
- Exterior features
- And many other property characteristics

## 🔧 Project Pipeline

The complete machine learning workflow was:

1. Data Loading
2. Data Understanding
3. Exploratory Data Analysis
4. Missing Value Analysis
5. Data Preprocessing
6. Feature Engineering
7. Numerical Feature Processing
8. Categorical Feature Encoding
9. Train/Validation Split
10. Baseline Model Training
11. Model Comparison
12. Hyperparameter Tuning
13. Final Model Selection
14. Stacking Ensemble
15. SHAP Explainability
16. Final Model Saving
17. Prediction Testing

## 🛠️ Feature Engineering

Feature engineering increased the feature space from:

- Original features: **80**
- Engineered features: **15**
- Final features before preprocessing: **95**

Important engineered/processed information included housing size, quality, age, renovation, and other property-related relationships.

After preprocessing:

- Numerical features: **52**
- Categorical features: **43**
- Processed features: **317**

## 🤖 Models Used

Three tree-based machine learning models were compared:

### Random Forest

Random Forest is an ensemble of decision trees where multiple trees are trained using random samples of data and features.

It reduces variance through averaging multiple trees.

### XGBoost

XGBoost is a gradient boosting algorithm that builds trees sequentially, where each new tree attempts to correct errors made by previous trees.

It was the strongest individual model in this project.

### LightGBM

LightGBM is a gradient boosting framework designed for efficient and fast training, especially on larger datasets.

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using 5-fold cross-validation.

### Tuned XGBoost

Best parameters:

- `subsample = 0.9`
- `reg_lambda = 1`
- `reg_alpha = 0.01`
- `n_estimators = 900`
- `min_child_weight = 3`
- `max_depth = 3`
- `learning_rate = 0.02`
- `colsample_bytree = 0.8`

The tuned XGBoost model achieved:

- **MAE: 15,718.05**
- **RMSE: 24,869.62**
- **R²: 0.9194**

## 🏆 Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Tuned Random Forest | 18,224.33 | 32,565.03 | 0.8617 |
| Tuned XGBoost | **15,718.05** | **24,869.62** | **0.9194** |
| Tuned LightGBM | 17,073.18 | 28,264.73 | 0.8958 |
| Stacking Model | 15,871.83 | 26,317.67 | 0.9097 |

### Final Model Selection

The **Tuned XGBoost model** was selected as the final production model because it achieved the best validation performance.

Although the stacking model was also tested, it did not outperform tuned XGBoost.

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) was used to understand how individual features influence predictions.

The most influential features included:

1. `OverallQual`
2. `GrLivArea`
3. `TotalBsmtSF`
4. `BsmtFinSF1`
5. `GarageCars`
6. `YearBuilt`
7. `YearRemodAdd`
8. `LotArea`

This provides interpretability instead of treating the model as a complete black box.

## 📈 Final Validation Performance

The final tuned XGBoost model achieved approximately:

- **MAE:** $15,718
- **RMSE:** $24,870
- **R²:** 0.919

An R² of approximately **0.92** means the model explains about 92% of the variance in house sale prices on the validation dataset.

The average absolute percentage error was approximately **9.18%**.

## 💾 Saved Model Files

The trained artifacts are included in the repository:

- `final_xgboost_model.joblib`
- `final_house_price_model.joblib`
- `preprocessor.joblib`
- `final_metrics.joblib`

The preprocessing object is saved so that new data can undergo the same transformations used during training.

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## ▶️ How to Run

Clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
