🚀 OncoVision: An Explainable Multi-Modal Breast Cancer Prediction System
Machine Learning + Deep Learning + Explainable AI for Clinical Decision Support
OncoVision is an integrated breast cancer prediction framework that combines clinical tabular data, mammogram imaging, and long-term risk forecasting into a single, explainable diagnostic system.
The system leverages ensemble ML models, CNN-based image classification, and explainable AI tools (SHAP, LIME, GradCAM) to deliver transparent, accurate, and clinically meaningful predictions.

This project is implemented in Python and deployed through a Streamlit-based web interface, enabling real-time, user-friendly diagnosis assistance.

⭐ Key Features

🔹 1. Tabular Clinical Data Prediction
Uses XGBoost and Random Forest ensemble models
Trained on Wisconsin Breast Cancer Dataset (WBCD)
Achieved 97.37% accuracy (XGBoost) and 96.49% (Random Forest)
Integrated Explainability:
SHAP → Global & local feature attribution
LIME → Instance-level interpretability

🔹 2. Mammogram Image Classification
CNN model trained on MIAS mammography dataset
Identifies benign vs. malignant masses
GradCAM heatmaps highlight suspicious regions
Achieved 94.12% accuracy

🔹 3. Long-Term Risk Forecasting
Predicts 1-year, 3-year, and 5-year survival probabilities
Multi-modal fusion of tabular + imaging features
Generates personalized risk curves for clinical insight

🔹 4. Fully Explainable Clinical AI
Our system integrates three layers of interpretability:
SHAP – Most influential clinical features (e.g., concave points, area, perimeter)
LIME – Per-patient sensitivity analysis
GradCAM – Visual heatmaps over mammograms

🔹 5. Web Deployment (Streamlit)
The application offers:
Image upload
Tabular input form
Real-time model inference
Visual explanations
Risk forecasting dashboard

📊 Model Performance Summary
Model	Accuracy	Precision	Recall	F1-score	AUC-ROC
XGBoost	97.37%	97.56%	95.24%	96.39%	0.9951
Random Forest	96.49%	95.24%	95.24%	95.24%	0.9923
CNN (MIAS)	94.12%	92.86%	91.67%	92.26%	0.9745

📁 Project Structure
OncoVision/
│── models/                # (Empty in GitHub – external download link below)
│── src/
│     ├── train_model.py
│     ├── predict_tabular.py
│     ├── predict_image.py
│     ├── explanation_shap.py
│     ├── explanation_lime.py
│     ├── gradcam.py
│── app/
│     ├── streamlit_app.py
│── requirements.txt
│── README.md


📥 Download Trained Model (.h5)
Due to GitHub’s 100MB file limit, trained models are stored externally.
👉 Download OncoVision CNN Model:
🔗 https://drive.google.com/file/d/1A6nI-h55LPt25Rl58wN0QlipD69s12Ad/view?usp=share_link
Place downloaded models in:
/models


🔧 Tech Stack
1. Python 3.11
2. Scikit-learn, XGBoost, PyTorch / TensorFlow , OpenCV, NumPy, Pandas
3. SHAP, LIME, GradCAM
4. Streamlit (UI)
5. Matplotlib / Seaborn (visualizations)
   

▶️ How to Run Locally
1. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
2. Install Requirements
pip install -r requirements.txt
3. Add Model Files
Download from Drive and place inside /models.
4. Run Streamlit App
streamlit run app/streamlit_app.py


🎯 Project Highlights
1. Multi-modal design mirrors real clinical workflows
2. Transparent, reliable ML predictions
3. Real-time diagnosis assistance
4. Performance competitive with published research
5. Bridges research → clinical usability gap

   
<img width="400px" height="auto" alt="mammo prediction" src="https://github.com/user-attachments/assets/9c442748-f0fb-478f-b340-ef5b9f90d87a" />


<img width="400px" height="auto" alt="image" src="https://github.com/user-attachments/assets/c53dadc8-1c76-491d-840f-6ce83ecc908f" />


<img width="400px" height="auto" alt="image" src="https://github.com/user-attachments/assets/80a74530-397f-444b-a38c-f7a8e57ec1bc" />

<img width="400px" height="auto" alt="image" src="https://github.com/user-attachments/assets/7a3a70ff-0916-4ebd-81e9-2f0b5a6533c2" />



🔮 Future Enhancements
1. Support for DICOM mammograms
2. Multi-center dataset validation
3. Integration with hospital EHR systems
4. Uncertainty estimation

