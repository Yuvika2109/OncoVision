import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF1493;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<p class="main-header">🎗️ Breast Cancer Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML with Explainable AI</p>', unsafe_allow_html=True)

# Introduction
st.write("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>📊 Tabular Prediction</h3>
        <p>Clinical feature analysis using XGBoost & Random Forest</p>
        <ul>
            <li>30 clinical features</li>
            <li>SHAP explanations</li>
            <li>LIME interpretability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>🖼️ Image Classification</h3>
        <p>CNN-based mammogram analysis</p>
        <ul>
            <li>Deep learning models</li>
            <li>GradCAM visualization</li>
            <li>Region highlighting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h3>📈 Risk Forecasting</h3>
        <p>5-year survival prediction</p>
        <ul>
            <li>Multi-modal fusion</li>
            <li>Longitudinal analysis</li>
            <li>Coming Soon!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# How to Use
st.header("📖 How to Use This Application")

st.markdown("""
1. **Navigate** using the sidebar to choose a prediction method
2. **Upload** your data or use sample data
3. **Get predictions** with confidence scores
4. **Explore** explainability visualizations
5. **Download** results for further analysis
""")

st.write("---")

# Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset Size", "569 samples", "WBCD")

with col2:
    st.metric("Features", "30", "Clinical")

with col3:
    st.metric("Images", "20,000+", "Mammograms")

with col4:
    st.metric("Model Accuracy", "95%+", "Average")

st.write("---")

# Footer
st.info("👈 **Select a feature from the sidebar to get started!**")

st.markdown("""
---
**Developed with ❤️ using Streamlit, TensorFlow, and XGBoost**

*Disclaimer: This is a demonstration project for educational purposes. 
Always consult healthcare professionals for medical decisions.*
""")