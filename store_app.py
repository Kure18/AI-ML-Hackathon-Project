import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping
import plotly.express as px

# Set up page configurations
st.set_page_config(page_title="Store Sales Analytics & Forecasting App", layout="wide")

st.title("📊 Store Sales Forecasting & Analytics Application")
st.markdown("""
This app processes sales datasets, executes a complete feature engineering pipeline, 
and uses an **Optimized LightGBM Regressor** to predict total sales.
""")

# --- SIDEBAR: DATA UPLOAD VIA DROPDOWNS ---
st.sidebar.header("📁 Upload Datasets")

# Dropdown to expand and upload Train File
with st.sidebar.expander("Step 1: Add Training Dataset", expanded=True):
    train_file = st.file_uploader("Choose a train CSV file", type=["csv"], key="train")

# Dropdown to expand and upload Test File
with st.sidebar.expander("Step 2: Add Testing Dataset", expanded=True):
    test_file = st.file_uploader("Choose a test CSV file", type=["csv"], key="test")

# --- MODEL CORE PIPELINE ---
def process_and_predict(train, test):
    # Progress Tracking
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Isolating Target Variable...")
    target = train['total_sales']
    train_features = train.drop(columns=['total_sales'])
    
    progress_bar.progress(15)
    status_text.text("Combining datasets for data consistency...")
    combined = pd.concat([train_features, test], ignore_index=True)
    
    # 1. Clean Text Columns
    combined['fat_content'] = combined['fat_content'].str.lower().str.strip()
    combined['product_category'] = combined['product_category'].str.lower().str.strip()
    
    # 2. Missing Value Imputation
    combined['product_weight_kg'] = combined.groupby('product_category')['product_weight_kg'].transform(lambda x: x.fillna(x.mean()))
    store_size_mode = combined.groupby('store_format', observed=False)['store_size'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Medium")
    combined['store_size'] = store_size_mode
    
    combined['shelf_visibility'] = combined['shelf_visibility'].replace(0.0, np.nan)
    combined['shelf_visibility'] = combined.groupby('product_code')['shelf_visibility'].transform(lambda x: x.fillna(x.mean()))
    combined['shelf_visibility'] = combined['shelf_visibility'].fillna(combined['shelf_visibility'].mean())
    
    # 3. Engineering New Prefix and Interaction Features
    combined['product_type_prefix'] = combined['product_code'].str.split('-').str[1]
    combined['store_code_prefix'] = combined['store_code'].str.split('-').str[1]
    
    mean_vis = combined.groupby('store_code')['shelf_visibility'].transform('mean')
    combined['visibility_store_average'] = combined['shelf_visibility'] / mean_vis
    
    tier_mapping = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}
    combined['store_location_tier'] = combined['store_location_tier'].map(tier_mapping)
    
    # 4. Enforce categorical datatypes
    categorical_col = ['product_type_prefix', 'store_code_prefix', 'product_category', 'fat_content', 'store_format', 'store_size']
    for col in categorical_col:
        combined[col] = combined[col].astype('category')
        
    progress_bar.progress(40)
    status_text.text("Splitting datasets & creating out-of-fold target encodings...")
    
    # Resplit Data
    train_df = combined.iloc[:len(train)].copy()
    test_df = combined.iloc[len(train):].copy()
    train_df['target_log'] = np.log1p(target)
    
    # 5. Out of Fold Target Encoding to avoid data leakage
    columns_to_encode = ['product_category', 'store_code']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in columns_to_encode:
        train_df[f'{col}_encoded'] = np.nan
        test_df[f'{col}_encoded'] = np.nan
        
        for train_idx, val_idx in kf.split(train_df):
            X_tr, X_va = train_df.iloc[train_idx], train_df.iloc[val_idx]
            means = X_tr.groupby(col, observed=False)['target_log'].mean()
            train_df.iloc[val_idx, train_df.columns.get_loc(f'{col}_encoded')] = X_va[col].map(means).astype(float)
            
        global_means = train_df.groupby(col, observed=False)['target_log'].mean()
        test_df[f'{col}_encoded'] = test_df[col].map(global_means).astype(float)
        
        train_df[f'{col}_encoded'] = train_df[f'{col}_encoded'].fillna(train_df['target_log'].mean())
        test_df[f'{col}_encoded'] = test_df[f'{col}_encoded'].fillna(train_df['target_log'].mean())

    # Drop Identifiers 
    drop_cols = ['id', 'product_code', 'store_code']
    X_train_full = train_df.drop(columns=drop_cols + ['target_log'])
    X_test = test_df.drop(columns=drop_cols)
    y_train_full_log = train_df['target_log']
    
    progress_bar.progress(65)
    status_text.text("Training final LightGBM Model with optimized parameters...")
    
    # 6. Train-Validation Split for Early Stopping Assessment
    X_train, X_val, y_train_log, y_val_log = train_test_split(X_train_full, y_train_full_log, test_size=0.2, random_state=42)
    
    # Hardcoded optimized hyperparameters generated from your Optuna study
    best_params = {
        'learning_rate': 0.016712714813575447, 
        'num_leaves': 16, 
        'min_child_samples': 30, 
        'feature_fraction': 0.6829837018688212, 
        'bagging_fraction': 0.6028472718704192, 
        'bagging_freq': 3,
        'n_estimators': 2000,
        'random_state': 42,
        'n_jobs': -1
    }
    
    final_model = LGBMRegressor(**best_params)
    final_model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[early_stopping(stopping_rounds=150, verbose=False)]
    )
    
    progress_bar.progress(85)
    status_text.text("Generating validations metrics and final test predictions...")
    
    # 7. Model evaluation metrics
    val_preds_log = final_model.predict(X_val)
    improved_log_rmse = np.sqrt(mean_squared_error(y_val_log, val_preds_log))
    
    val_preds_real = np.expm1(val_preds_log)
    y_val_real = np.expm1(y_val_log)
    improved_real_rmse = np.sqrt(mean_squared_error(y_val_real, val_preds_real))
    
    # 8. Unseen test asset predictions
    test_preds_log = final_model.predict(X_test)
    test_preds_real = np.clip(np.expm1(test_preds_log), 0, None)
    
    # Format and aggregate predictions by key groups
    store_sales_predictions = pd.DataFrame({
        'product_code': test['product_code'],
        'store_code': test['store_code'],
        'total_sales': test_preds_real
    })
    final_predictions = store_sales_predictions.groupby(['product_code', 'store_code'])['total_sales'].sum().reset_index()
    
    progress_bar.progress(100)
    status_text.empty()
    
    return final_predictions, improved_log_rmse, improved_real_rmse

# --- MAIN APP LAYOUT CONTROLLER ---
if train_file is not None and test_file is not None:
    # Read Files
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Show Summary Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Data Overview", "🔮 Run Model Predictions", "📊 Sales Insights"])
    
    with tab1:
        st.subheader("Initial Raw Data Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Training Set:** `{df_train.shape[0]}` rows | `{df_train.shape[1]}` columns")
            st.dataframe(df_train.head(10), use_container_width=True)
        with col2:
            st.markdown(f"**Testing Set:** `{df_test.shape[0]}` rows | `{df_test.shape[1]}` columns")
            st.dataframe(df_test.head(10), use_container_width=True)
            
    with tab2:
        st.subheader("Model Execution Center")
        if st.button("🚀 Process Data & Generate Predictions"):
            with st.spinner("Executing pipeline tasks... Please stand by."):
                preds_df, log_rmse, real_rmse = process_and_predict(df_train, df_test)
                
            st.success("Analysis Complete!")
            
            # Show Metrics
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Optimized Validation Log RMSE", f"{log_rmse:.5f}")
            m_col2.metric("Real-World Scaled Validation RMSE", f"{real_rmse:.2f} Units")
            
            # Display Forecast Table
            st.write("### Predicted Total Sales Output Table")
            st.dataframe(preds_df, use_container_width=True)
            
            # File Exporter Button
            csv_data = preds_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download 'final_predictions_optimized.csv'",
                data=csv_data,
                file_name="final_predictions_optimized.csv",
                mime="text/csv"
            )
            
    with tab3:
        st.subheader("Exploratory Distributions (Training Set)")
        # Plot distribution of original target variable if available
        if 'total_sales' in df_train.columns:
            fig = px.histogram(df_train, x='total_sales', nbins=50, title="Distribution of Total Sales", color_discrete_sequence=['#4A90E2'])
            st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.box(df_train, x='store_format', y='product_price', title="Product Price Range by Store Format", color='store_format')
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("💡 Please upload **both** a training and testing CSV dataset via the sidebar expanders to begin processing.")