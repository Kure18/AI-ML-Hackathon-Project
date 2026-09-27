import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="DSN Mart Sales Forecasting Dashboard",
    page_icon="🏪",
    layout="wide"
)

# Title and introduction
st.title("🏪 DSN Mart Sales Forecasting Model Hub")
st.markdown("""
This application details the technical blueprint, architectural framework, and operational mechanics of the **Ensemble Machine Learning Pipeline** optimized to forecast total store sales at DSN Mart.
""")

# Sidebar Navigation
st.sidebar.header("Pipeline Navigation")
page = st.sidebar.radio(
    "Select a Pipeline Stage:",
    ["Overview & Dataset Info", "Data Preprocessing & Cleaning", "Feature Engineering", "Model Architecture & Weights", "Interactive Inference Demo"]
)

# Define static facts from the notebook metadata
categorical_cols = ['product_type_prefix', 'store_code_prefix', 'product_category', 'fat_content', 'store_format', 'store_size']
meta_weights = {'LightGBM': 0.2459, 'XGBoost': 0.2119, 'CatBoost': 0.5663}

# --- PAGE 1: OVERVIEW ---
if page == "Overview & Dataset Info":
    st.header("📋 Pipeline Overview")
    st.write("""
    The sales forecasting system uses a **5-Fold Cross-Validation Ensembled Matrix Stacker**. 
    It combines tree-based gradient boosted models with a robust out-of-fold target encoding layer to capture micro-market density and price elasticity without introducing data leakage.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Features Handled")
        st.markdown("""
        * **Product Metrics**: Code, category, weight, fat content, price, and shelf visibility.
        * **Store Dynamics**: Code, format, size, location tier, and product assortment density.
        * **Target Variable**: `total_sales` (Modeled using a scale-invariant Log1p transformation).
        """)
    with col2:
        st.subheader("Engineering Workflow Stack")
        st.info("🛠️ Data Wrangling: Pandas & NumPy\n\n📉 Cross-Validation: Scikit-learn KFold (5 Splits)\n\n🤖 ML Engines: LightGBM, XGBoost, CatBoost\n\n⚙️ Stacking Meta-Learner: Positive Ridge Regression")

# --- PAGE 2: PREPROCESSING ---
elif page == "Data Preprocessing & Cleaning":
    st.header("🧽 Data Cleaning & Imputation Protocol")
    st.write("To prevent global mean skewing, missing values and anomalies are imputed using conditional distribution logic:")
    
    st.code("""
# 1. Standardizing Categorical Strings
combined['fat_content'] = combined['fat_content'].str.lower().str.strip()
combined['product_category'] = combined['product_category'].str.lower().str.strip()

# 2. Category-Specific Feature Imputation
combined['product_weight_kg'] = combined.groupby('product_category')['product_weight_kg'].transform(lambda x: x.fillna(x.mean()))
store_size_mode = combined.groupby('store_format')['store_size'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Medium")
combined['store_size'] = store_size_mode

# 3. Handling Zero Value Anomalies in Visibility
combined['shelf_visibility'] = combined['shelf_visibility'].replace(0.0, np.nan)
combined['shelf_visibility'] = combined.groupby('product_code')['shelf_visibility'].transform(lambda x: x.fillna(x.mean()))
combined['shelf_visibility'] = combined['shelf_visibility'].fillna(combined['shelf_visibility'].mean())
    """, language="python")
    
    st.success("✔️ Imputation successfully prevents data leakage by pinning metrics back to localized grouping frames.")

# --- PAGE 3: FEATURE ENGINEERING ---
elif page == "Feature Engineering":
    st.header("🏗️ Advanced Feature Engineering & Leakage-Free Encoding")
    
    tab1, tab2 = st.tabs(["Interaction Ratios & Keys", "Out-of-Fold Target Encoding"])
    
    with tab1:
        st.markdown("### Pricing Elasticity & Interaction Ratios")
        st.write("Extracted hidden structural pricing signals relative to individual markets:")
        st.code("""
# Structural key extraction
combined['product_type_prefix'] = combined['product_code'].str.split('-').str[1]
combined['store_code_prefix'] = combined['store_code'].str.split('-').str[1]

# Pricing ratios
combined['price_to_weight'] = combined['product_price'] / (combined['product_weight_kg'] + 0.1)
combined['store_mean_price'] = combined.groupby('store_code')['product_price'].transform('mean')
combined['price_relative_to_store'] = combined['product_price'] / combined['store_mean_price']

# Visibility scaling & Micro-market density
combined['visibility_store_average'] = combined['shelf_visibility'] / combined.groupby('store_code')['shelf_visibility'].transform('mean')
combined['visibility_to_cat_average'] = combined['shelf_visibility'] / combined.groupby('product_category')['shelf_visibility'].transform('mean')
combined['store_product_count'] = combined.groupby('store_code')['product_code'].transform('count')
        """, language="python")
        
    with tab2:
        st.markdown("### 5-Fold Leakage-Free Target Encoding")
        st.write("High cardinality categorical parameters (`product_category` and `store_code`) are target-encoded within an out-of-fold configuration using log-transformed sales values:")
        st.code("""
# Target encoding computed strictly within fold constraints to avoid target leakage
for train_idx, val_idx in kf.split(train_df):
    X_tr, X_va = train_df.iloc[train_idx], train_df.iloc[val_idx]
    for col in columns_to_encode:
        means = X_tr.groupby(col, observed=False)['target_log'].mean()
        train_df.iloc[val_idx, train_df.columns.get_loc(f'{col}_encoded')] = X_va[col].map(means).astype(float)
        
# Map untainted global means back to test dataset structures
        """, language="python")

# --- PAGE 4: MODEL ARCHITECTURE ---
elif page == "Model Architecture & Weights":
    st.header("🤖 Ensembled Model Stacking Architecture")
    st.write("Predictions are generated through three distinct algorithmic layouts, blended dynamically by a Meta-Learning **Ridge Regression Stacker** forcing positive coefficient constraints.")
    
    # Render weights bar chart
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ['#4A90E2', '#50E3C2', '#B8E986']
    bars = ax.barh(list(meta_weights.keys()), list(meta_weights.values()), color=colors, height=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Meta-Weight Multiplier Coefficient")
    ax.set_title("Optimized Model Blending Contribution")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center', ha='left', fontweight='bold')
    st.pyplot(fig)

    st.markdown("### Base Learner Configurations")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**A. LightGBM**")
        st.caption("Native structural categorical tree parsing.")
        st.json({"learning_rate": 0.01256, "num_leaves": 15, "min_child_samples": 47, "feature_fraction": 0.7947, "bagging_fraction": 0.6546})
    with col2:
        st.markdown("**B. XGBoost**")
        st.caption("Processes dummy matrix one-hot encoded alignments.")
        st.json({"learning_rate": 0.015, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8, "early_stopping_rounds": 150})
    with col3:
        st.markdown("**C. CatBoost**")
        st.caption("Phenomenal handling of raw string category factors.")
        st.json({"iterations": 3000, "learning_rate": 0.02, "depth": 5, "eval_metric": "RMSE"})

# --- PAGE 5: INTERACTIVE INFERENCE ---
elif page == "Interactive Inference Demo":
    st.header("🔮 Simulated Live Store Forecast Tool")
    st.write("Adjust features to generate dynamic ensembled predictions based on your optimized model rules:")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        prod_price = c1.number_input("Product Retail Price ($)", min_value=1.0, max_value=500.0, value=140.0)
        prod_weight = c1.number_input("Product Weight (kg)", min_value=0.05, max_value=50.0, value=12.5)
        shelf_vis = c1.slider("Shelf Visibility Score", 0.001, 0.350, 0.06)
    with c2:
        store_format = c2.selectbox("Store Format", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
        store_size = c2.selectbox("Store Size Tier", ["Small", "Medium", "High"])
        store_count = c2.number_input("Store Assortment Product Count", value=930)
    with c3:
        cat_encoded = c3.slider("OOF Product Category Encoded Log-Mean", 5.0, 10.0, 7.2)
        store_encoded = c3.slider("OOF Store Code Encoded Log-Mean", 5.0, 10.0, 7.8)
    
    # Calculate inferred feature interactions
    price_to_weight = prod_price / (prod_weight + 0.1)
    store_mean_price_sim = 142.0
    price_relative = prod_price / store_mean_price_sim
    
    if st.button("⚡ Calculate Composite Forecast Metrics"):
        # Synthesize baseline simulation predictions across the log target spectrum
        sim_log_lgb = (cat_encoded * 0.4) + (store_encoded * 0.6) + (price_relative * 0.2)
        sim_log_xgb = (cat_encoded * 0.35) + (store_encoded * 0.65) + (price_relative * 0.15)
        sim_log_cat = (cat_encoded * 0.42) + (store_encoded * 0.58) + (price_relative * 0.22)
        
        # Apply Stacker Weights
        final_log_pred = (sim_log_lgb * meta_weights['LightGBM']) + \
                         (sim_log_xgb * meta_weights['XGBoost']) + \
                         (sim_log_cat * meta_weights['CatBoost'])
        
        # Inverse log transform to convert back to currency units
        final_sales_real = np.expm1(final_log_pred)
        
        st.markdown("---")
        st.subheader("🎯 Resulting Predictions Matrix")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted Total Sales Value", f"${final_sales_real:,.2f}")
        res_col2.metric("Blended Log1p Target Scale Index", f"{final_log_pred:.4f}")
        

        
        

                    


