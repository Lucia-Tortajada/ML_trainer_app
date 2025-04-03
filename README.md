# 🤖 ML Model Trainer App

This Streamlit application allows users to interactively train machine learning models on preloaded datasets or their own uploaded CSV files. The goal of this project was to build a flexible and user-friendly interface for training, evaluating, and visualizing machine learning models .

---

## 🚀 Features

- 📂 **Dataset input**: Choose from Seaborn sample datasets or upload your own CSV
- ✍️ **Feature selection**: Easily select numerical and categorical features and the target variable
- ⚙️ **Model configuration**: 
  - Train **regression or classification models**
  - Select and tune models like Linear Regression, Random Forest Regressor, Random Forest Classifier
  - Adjust test size and hyperparameters (e.g. number of trees)
- 📊 **Visualizations**:
  - Residual distribution (for regression)
  - Confusion matrix (for classification)
  - ROC curve (for binary classification)
  - Feature importance (for tree-based models)
- 💾 **Export**: Download trained models and datasets
- 🔄 **Session State & Forms**: Parameters are only submitted when the user clicks "Fit model" to avoid accidental training
- ✅ Meets all basic, intermediate, and advanced requirements from assignment

---

## 🧠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Seaborn & Matplotlib
- Pandas & NumPy

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Lucia-Tortajada/ML_trainer_app.git
cd ML_trainer_app

### 2. Create and activate a virtual environment

# On Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the app

streamlit run app.py
