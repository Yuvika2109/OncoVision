import streamlit as st
import pandas as pd

st.set_page_config(page_title="About", page_icon="📖", layout="wide")

st.title("📖 About This Project")

st.markdown("""
## 🎗️ Breast Cancer Prediction with ML & Explainable AI

This comprehensive system demonstrates advanced machine learning techniques combined with explainable AI
for breast cancer detection and risk assessment using multiple data modalities.
""")

st.write("---")

# Project Features
st.header("✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Tabular Prediction
    **Clinical Feature Analysis**
    - XGBoost & Random Forest models
    - 569 patient samples
    - 30 clinical features
    - SHAP & LIME explanations
    - 95%+ accuracy
    
    **Use Case:**
    Early detection using biopsy features
    """)

with col2:
    st.markdown("""
    ### 🖼️ Image Classification
    **CNN-based Mammogram Analysis**
    - Deep learning architecture
    - MIAS dataset (322 real images)
    - GradCAM visualization
    - 92.8% accuracy
    
    **Use Case:**
    Automated mammogram screening
    """)

with col3:
    st.markdown("""
    ### 📈 Risk Forecasting
    **5-Year Risk Prediction**
    - Gradient Boosting + Random Forest
    - Multi-modal data fusion
    - Temporal risk projection
    - Personalized recommendations
    
    **Use Case:**
    Long-term risk assessment
    """)

st.write("---")

# Technology Stack
st.header("🛠️ Technology Stack")

tech_data = pd.DataFrame({
    'Category': ['Frontend', 'ML/DL Framework', 'Models', 'Explainability', 'Visualization', 'Data Processing'],
    'Technologies': [
        'Streamlit',
        'TensorFlow, Keras, Scikit-learn',
        'XGBoost, Random Forest, CNN, Gradient Boosting',
        'SHAP, LIME, GradCAM',
        'Plotly, Matplotlib, Seaborn',
        'Pandas, NumPy, OpenCV'
    ]
})

st.table(tech_data)

st.write("---")

# Datasets
st.header("📊 Datasets Used")

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

with dataset_col1:
    st.markdown("""
    **Wisconsin Breast Cancer**
    - Source: UCI ML Repository
    - Samples: 569 patients
    - Features: 30 measurements
    - Classes: Benign/Malignant
    - Type: Tabular CSV
    """)

with dataset_col2:
    st.markdown("""
    **MIAS Mammograms**
    - Source: UK Medical Database
    - Images: 322 mammograms
    - Format: Grayscale PNG
    - Resolution: 224×224
    - Type: Medical imaging
    """)

with dataset_col3:
    st.markdown("""
    **Risk Factor Data**
    - Clinical history
    - Demographics
    - Lifestyle factors
    - Screening records
    - Type: Multi-modal
    """)

st.write("---")

# Model Performance
st.header("🎯 Model Performance Summary")

perf_data = pd.DataFrame({
    'Module': ['Tabular Prediction', 'Image Classification', 'Risk Forecasting'],
    'Primary Model': ['XGBoost', 'CNN', 'Gradient Boosting'],
    'Accuracy': ['95.6%', '92.8%', '94.2%'],
    'Key Feature': ['SHAP Explanation', 'GradCAM Heatmap', '5-Year Projection']
})

st.table(perf_data)

st.write("---")

# Explainability
st.header("🔍 Explainable AI Techniques")

explain_col1, explain_col2 = st.columns(2)

with explain_col1:
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)**
    - Game theory-based approach
    - Feature contribution calculation
    - Global and local explanations
    - Used in: Tabular Prediction & Risk Forecasting
    
    **LIME (Local Interpretable Model-agnostic Explanations)**
    - Local surrogate models
    - Perturbation-based analysis
    - Model-agnostic approach
    - Used in: Tabular Prediction
    """)

with explain_col2:
    st.markdown("""
    **GradCAM (Gradient-weighted Class Activation Mapping)**
    - Gradient-based visualization
    - Convolutional layer analysis
    - Spatial localization of features
    - Used in: Image Classification
    
    **Feature Importance**
    - Tree-based importance scores
    - Temporal risk factor analysis
    - Population comparison
    - Used in: Risk Forecasting
    """)

st.write("---")

# Use Cases
st.header("💡 Clinical Use Cases")

use_case_col1, use_case_col2 = st.columns(2)

with use_case_col1:
    st.markdown("""
    **For Clinicians:**
    - 🏥 Assist in diagnostic decision-making
    - 📋 Interpret biopsy results
    - 🔬 Analyze mammogram images
    - 📊 Assess long-term patient risk
    - 📈 Plan screening schedules
    """)

with use_case_col2:
    st.markdown("""
    **For Patients:**
    - 📱 Understand diagnosis results
    - 🎯 Know personal risk level
    - 📅 Plan preventive measures
    - 💪 Make informed lifestyle choices
    - 🩺 Track screening compliance
    """)

st.write("---")

# Future Enhancements
st.header("🚀 Future Enhancements")

st.markdown("""
### Planned Features:

**Phase 2:**
- 🔄 Real-time model updates with new data
- 📱 Mobile application deployment
- 🌐 Cloud-based API endpoints
- 📊 Interactive dashboards for clinicians
- 🔐 HIPAA-compliant data handling

**Phase 3:**
- 🤖 Multi-cancer detection (lung, colon, etc.)
- 🧬 Genetic risk factor integration
- 📈 Real-time monitoring systems
- 🏥 EHR system integration
- 🌍 Multi-language support

**Phase 4:**
- 🧪 Clinical trial integration
- 📱 Wearable device data fusion
- 🤝 Collaborative diagnostic platform
- 📚 Continuous learning system
- ✅ FDA/CE certification pathway
""")

st.write("---")

# Team & Contact
st.header("👥 Project Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    **Project Details:**
    - **Project Name:** Breast Cancer ML with Explainable AI
    - **Version:** 1.0.0
    - **Date:** November 2024
    - **Framework:** TensorFlow + Streamlit
    - **License:** Educational Use
    """)

with info_col2:
    st.markdown("""
    **Key Technologies:**
    - Machine Learning
    - Deep Learning
    - Explainable AI
    - Medical Imaging
    - Risk Assessment
    """)

st.write("---")

# Disclaimer
st.header("⚠️ Important Disclaimer")

st.error("""
**Medical Disclaimer:**

This application is developed for **educational and research purposes only**.

**Important Points:**
- ❌ NOT approved for clinical use
- ❌ NOT a substitute for professional medical advice
- ❌ NOT validated by regulatory bodies (FDA/CE)
- ✅ For demonstration and learning purposes
- ✅ Requires expert medical interpretation
- ✅ Should be used alongside clinical judgment

**Always consult qualified healthcare professionals (radiologists, oncologists, pathologists) 
for actual medical diagnosis, treatment decisions, and patient care.**

**This system has not undergone clinical validation or regulatory approval.**
""")

st.write("---")

# References
st.header("📚 References & Resources")

st.markdown("""
### Datasets:
1. [UCI ML Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
2. [MIAS - Mammographic Image Analysis Society](http://peipa.essex.ac.uk/info/mias.html)

### Research Papers:
1. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NIPS.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" KDD.
3. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.

### Libraries & Frameworks:
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Medical Resources:
- [American Cancer Society](https://www.cancer.org/)
- [National Cancer Institute](https://www.cancer.gov/)
- [WHO Cancer Information](https://www.who.int/cancer)
""")

st.write("---")

st.info("""
💡 **Acknowledgments:**
This project demonstrates the integration of machine learning, deep learning, and explainable AI
for breast cancer detection and risk assessment. Special thanks to the open-source community
and medical imaging researchers who made these datasets publicly available.
""")

st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 20px;'>
    <strong style='font-size: 16px;'>🎗️ Breast Cancer Detection & Risk Assessment System</strong><br>
    <span style='font-size: 12px; color: #666;'>
        Powered by Machine Learning | Explained by AI | Built for Healthcare
    </span>
</div>
""", unsafe_allow_html=True)