# Chronic Disease Prediction with Streamlit

This project focuses on predicting chronic diseases such as diabetes or heart disease using machine learning, with a Streamlit demo app for deployment.

## 🧠 Features
- Use of structured data (e.g., UCI dataset)
- ML models: Logistic Regression, Random Forest
- SHAP for interpretability
- Streamlit UI for user interaction

## 🗂️ Project Structure
- `data/` – Input datasets
- `notebooks/` – Jupyter Notebooks for EDA, modeling
- `models/` – Trained ML models (.pkl)
- `streamlit_app/` – Streamlit front-end app
- `results/` – Charts and outputs

## 🚀 Running the Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
