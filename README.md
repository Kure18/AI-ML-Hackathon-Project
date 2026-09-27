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
## Model Evaluation & Ensemble Architecture

The modeling strategy leverages a robust **5-Fold Cross-Validation (Out-of-Fold)** stacking architecture [1] to minimize target leakages and ensure optimal structural generalization. The target metric (`total_sales`) undergoes a log-transformation (`np.log1p`) [1] during training to penalize proportional skewness safely.

### 1. Model Selection & Hyperparameters

Three distinct gradient boosting architectures are trained across each fold:

*   **LightGBM (`LGBMRegressor`)** [1]
    *   Optimized parameters: `learning_rate: 0.01256`, `num_leaves: 15`, `min_child_samples: 47`, `feature_fraction: 0.7947`, `bagging_fraction: 0.6546`.
    *   Handles natively encoded pandas categorical distributions.
*   **XGBoost (`XGBRegressor`)** [1]
    *   Optimized parameters: `learning_rate: 0.015`, `max_depth: 4`, `subsample: 0.8`, `colsample_bytree: 0.8`.
    *   Trained on explicit One-Hot encoded (`pd.get_dummies`) structural frames.
*   **CatBoost (`CatBoostRegressor`)** [1]
    *   Optimized parameters: `learning_rate: 0.02`, `depth: 5`, `eval_metric: 'RMSE'`.
    *   Leverages native object string mappings directly within the loop execution.

---

### 2. Meta-Learning Stacker Evaluation

To blend individual predictions efficiently without risk of over-fitting, a **Ridge Regression** meta-model is fitted over the Out-of-Fold (OOF) validation projections matrix. Positive constraints are enforced (`positive=True`) [1] to guarantee stable weighting allocations.

#### Optimized Blending Weights

The meta-learner automatically converged onto the following relative architectural coefficients:

| Model Framework | Stacking Coefficient (Weight) | Key Behavioral Strength |
| :--- | :--- | :--- |
| **CatBoost** [1] | **0.5663** [1] | Exceptional handling of raw un-encoded object strings as category factors. |
| **LightGBM** [1] | **0.2459** [1] | Rapid gradient optimization using deep histogram-based splits. |
| **XGBoost** [1] | **0.2119** [1] | Strong regularization on high-dimensional dummy-encoded columns. |

---

### 3. Pipeline Execution Validation




## Model Analysis
 ## Deep Model Analysis & Pipeline Mechanics

This section provides an architectural breakdown of the data pipelines, leakage prevention frameworks, and engineering methodologies utilized to solve the sales forecasting task.

### 1. Robust Leakage-Free Target Encoding
To protect the model from out-of-fold target leakages while encoding high-cardinality categorical variables (`product_category` and `store_code`), the pipeline executes a strict conditional mapping loop within a **5-Fold Cross-Validation layout**:
* **Isolation Framework:** Group-based conditional targets (`target_log`) are calculated exclusively using the training split indices of that specific fold (`X_tr`).
* **Out-of-Fold Allocation:** The values are mapped only onto the unseen validation fold indices (`val_idx`), preventing the global target mean from bleeding into training data.
* **Untainted Baselines:** Test sets are mapped exclusively using global statistics derived from the clean `train_df` frame. Missing categories are filled with the global target log mean as a structural safety net.

---

### 2. Feature Engineering & Structural Interaction Mechanics

The model's strong performance is largely driven by capturing market micro-densities, pricing elasticities, and store hierarchies.



# Key Observations are
## 🔍 Key Observations & Architectural Insights

This section outlines the critical data behaviors, model responses, and operational discoveries noticed during the pipeline's cross-validation and blending phases.

### 1. CatBoost Dominance in Categorical Processing
* **String Modeling Superiority:** `CatBoostRegressor` achieved the highest weight allocation (**0.5663**) from the Ridge meta-learner. This proves its framework handles raw, un-encoded object strings as category factors far better than traditional one-hot matrices.
* **Reduction in Feature Explosion:** By processing categories natively, CatBoost avoided the high-dimensional column expansion that one-hot encoding forced onto XGBoost. This preserved local patterns without diluting model focus.

---

### 2. Ensemble Diversity & Meta-Learner Synergies
* **Positive Constraint Balance:** Forcing positive constraints (`positive=True`) on the Ridge meta-model revealed a well-distributed ensemble structure. Instead of letting one framework take over entirely, it combined all three models effectively:
  \[\text{Final Prediction} = (0.5663 \times \text{Cat}) + (0.2459 \times \text{LGB}) + (0.2119 \times \text{XGB})\]
* **Complementary Framework Designs:** LightGBM's fast histogram-based splits and XGBoost's deep regularized leaves balanced out CatBoost's symmetric trees. This combination successfully reduced overall prediction variance.

---

### 3. Data Anomalies & Transformation Safeguards



# Data Source
## The data was sorted out from-DSN Bootcamp Qualification Hackathon 2026 ML Track | Kaggle 

# To view the DEPLOYED MODEL on my github account-https://github.com/Kure18/AI-ML-Hackathon-Project

