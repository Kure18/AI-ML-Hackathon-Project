# Predicting Total Sales for a Given Product at a Given DSN Mart Stores Across Nigeria
## Project Overview
### This project aims at a means of providing logistics for the sales of products at DSN Mart 
# Problem Definition :
## To know the driving force behind sales across DSN Mart Stores across the country
# Problem Solution :
## It provides solution to retail inventory and financial forecasting problems by predicting total sales for specific products across various retail stores 
# Business Value :
## - One of the key business value are optimization of revenue by maximizing shelf value of products and targeted pricing strategy for product price and store location tier
## - Aids stock reduction and logistics by minimizing stock out and overstocking of items and optimizing supply chain logistics.
## - Another business value is strategic growth by analysing performance trend across different variables such as store_size,store_format which provides a blueprint for future retail outlet.
# Business Metrics :
## The business metrics for the project to be successful are:
### - Sales and revenue
### - Operational and inventory efficiency
### - Store and space optimization
# Model Evaluation Results
## 📊 Model Performance & Hyperparameters

The model was optimized using **Optuna** over 30 trials with 5-fold cross-validation, targeting the minimization of the Log RMSE score. 

### Performance Metrics

| Metric | Target Scale | Value |
| :--- | :--- | :--- |
| **Validation Log RMSE** | Log-transformed (`log1p`) | **0.5166** |
| **Validation RMSE** | Real-world units (sales) | **1135.28** |

### Hyperparameters (LightGBM Regressor)

The final model was trained with early stopping (150 rounds) up to 2,000 estimators using these optimized parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `learning_rate` | `0.016712714813575447` | Step size shrinkage to prevent overfitting |
| `num_leaves` | `16` | Maximum tree leaves for base learners |
| `min_child_samples` | `30` | Minimum number of data needed in a child (leaf) |
| `feature_fraction` | `0.6829837018688212` | LightGBM will randomly select 68.3% of features |
| `bagging_fraction` | `0.6028472718704192` | LightGBM will randomly select 60.3% of the data |
| `bagging_freq` | `3` | Frequency for bagging (k-fold sub-sampling) |
| `random_state` | `42` | Seed for reproducibility |


## Model Analysis
Since we used the LightGBM regressor model with an RMSE of 0.516 we would deduce that the performance of the model is fair and our unscaled real world RMSE is 1135.28. The model could had performed much better if our dataset had a detailed information about the time that sales occur that is for it to be further broken into months,days,weeks and hours but rather it was in years. 

# Key Observations are
1. The final optimized model achieves an average prediction error of 0.5166 on the logarithmic scale (Optimized Validation Log RMSE).
2. When converted back to actual sales numbers, the model's predictions typically miss the real-world sales figures by about 1,135 units (Optimized Real World Scaled Validation RMSE of 1,135.28).
3. The hyperparameter tuning was highly successful, with Optuna trying 30 different parameter combinations to minimize error, eventually settling on a best trial score of roughly 0.5221 before final validation.
4. The model stopped training early at iteration 273 out of a possible 2,000 because its performance on the validation data stopped improving, effectively protecting it from overfitting.

# Data Source
## The data was sorted out from-DSN Bootcamp Qualification Hackathon 2026 ML Track | Kaggle 

