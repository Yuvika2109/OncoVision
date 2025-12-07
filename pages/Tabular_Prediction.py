import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
from lime.lime_tabular import LimeTabularExplainer
import plotly.graph_objects as go

st.set_page_config(page_title="Tabular Prediction", page_icon="📊", layout="wide")

st.title("📊 Tabular Prediction: Clinical Features")
st.markdown("### Predict breast cancer using clinical features with ML models")

# Sidebar
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "Random Forest", "Both"])
show_shap = st.sidebar.checkbox("Show SHAP Explanations", value=True)
show_lime = st.sidebar.checkbox("Show LIME Explanations", value=False)

# Load and prepare data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names, data.target_names

@st.cache_resource
def train_models(X_train, y_train):
    # Train XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    return xgb_model, rf_model

# Load data
df, feature_names, target_names = load_data()

# Display dataset info
with st.expander("📋 Dataset Information"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        st.metric("Classes", len(target_names))
    
    st.dataframe(df.head())

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train models
with st.spinner("Training models..."):
    xgb_model, rf_model = train_models(X_train, y_train)

st.success("✅ Models trained successfully!")

# Model performance
col1, col2 = st.columns(2)

with col1:
    xgb_score = xgb_model.score(X_test, y_test)
    st.metric("XGBoost Accuracy", f"{xgb_score*100:.2f}%")

with col2:
    rf_score = rf_model.score(X_test, y_test)
    st.metric("Random Forest Accuracy", f"{rf_score*100:.2f}%")

st.write("---")

# Prediction Section
st.header("🔮 Make Predictions")

input_method = st.radio("Choose Input Method:", ["Use Sample Data", "Manual Input"])

if input_method == "Use Sample Data":
    sample_idx = st.slider("Select Sample Index", 0, len(X_test)-1, 0)
    input_data = X_test[sample_idx].reshape(1, -1)
    actual_label = "Malignant" if y_test.iloc[sample_idx] == 0 else "Benign"
    st.info(f"Actual Label: **{actual_label}**")
else:
    st.write("Enter clinical feature values (normalized):")
    cols = st.columns(3)
    input_values = []
    
    for idx, feature in enumerate(feature_names):
        with cols[idx % 3]:
            value = st.number_input(
                feature[:20] + "...",
                value=0.0,
                format="%.4f",
                key=f"input_{idx}"
            )
            input_values.append(value)
    
    input_data = np.array(input_values).reshape(1, -1)

# Predict button
if st.button("🎯 Predict", type="primary"):
    st.write("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    # XGBoost Prediction
    if model_choice in ["XGBoost", "Both"]:
        with col1:
            st.markdown("#### XGBoost Model")
            xgb_pred = xgb_model.predict(input_data)[0]
            xgb_proba = xgb_model.predict_proba(input_data)[0]
            
            result = "Benign" if xgb_pred == 1 else "Malignant"
            color = "green" if xgb_pred == 1 else "red"
            
            st.markdown(f"<h2 style='color: {color};'>{result}</h2>", unsafe_allow_html=True)
            
            st.write("**Confidence Scores:**")
            st.write(f"- Malignant: {xgb_proba[0]*100:.2f}%")
            st.write(f"- Benign: {xgb_proba[1]*100:.2f}%")
            
            # Progress bars (fixed for compatibility)
            st.write("Malignant:")
            st.progress(float(xgb_proba[0]))
            st.write("Benign:")
            st.progress(float(xgb_proba[1]))
    
    # Random Forest Prediction
    if model_choice in ["Random Forest", "Both"]:
        with col2:
            st.markdown("#### Random Forest Model")
            rf_pred = rf_model.predict(input_data)[0]
            rf_proba = rf_model.predict_proba(input_data)[0]
            
            result = "Benign" if rf_pred == 1 else "Malignant"
            color = "green" if rf_pred == 1 else "red"
            
            st.markdown(f"<h2 style='color: {color};'>{result}</h2>", unsafe_allow_html=True)
            
            st.write("**Confidence Scores:**")
            st.write(f"- Malignant: {rf_proba[0]*100:.2f}%")
            st.write(f"- Benign: {rf_proba[1]*100:.2f}%")
            
            # Progress bars (fixed for compatibility)
            st.write("Malignant:")
            st.progress(float(rf_proba[0]))
            st.write("Benign:")
            st.progress(float(rf_proba[1]))
    
    st.write("---")
    
    # SHAP Explanations
    if show_shap:
        st.subheader("🔍 SHAP Explanations")
        st.write("Understanding which features contributed to the prediction:")
        
        try:
            with st.spinner("Generating SHAP explanations..."):
                # Use a smaller background dataset for faster computation
                background = shap.sample(X_train, 100)
                explainer = shap.KernelExplainer(xgb_model.predict_proba, background)
                shap_values = explainer.shap_values(input_data)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values[1],
                    input_data,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False
                )
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP visualization unavailable: {str(e)}")
    
    # LIME Explanations
    if show_lime:
        st.subheader("🔍 LIME Explanations")
        st.write("Local interpretable model-agnostic explanations:")
        
        try:
            with st.spinner("Generating LIME explanations..."):
                explainer = LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=target_names,
                    mode='classification'
                )
                
                exp = explainer.explain_instance(
                    input_data[0],
                    xgb_model.predict_proba,
                    num_features=10
                )
                
                # Display LIME explanation
                st.write("**Top Contributing Features:**")
                lime_df = pd.DataFrame(exp.as_list(), columns=['Feature', 'Contribution'])
                st.dataframe(lime_df)
                
                # Visualize
                fig, ax = plt.subplots(figsize=(10, 6))
                exp.as_pyplot_figure()
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"LIME visualization unavailable: {str(e)}")

st.write("---")
st.info("💡 **Tip**: Try different samples and compare predictions from both models!")