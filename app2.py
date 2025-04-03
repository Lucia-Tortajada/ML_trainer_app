#--STEP 1-- Import all libraries
import streamlit as st #Used to build the app
import pandas as pd #For the dataframe
import seaborn as sns #For the sample Datasets
import numpy as np
import matplotlib.pyplot as plt #To plot the graphs
import joblib #ADVANCED FEATURE : To export the model
import io #ADVANCED FEATURE : To download the model

#To build the model and evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve

#--STEP 2--General configuration of the page
st.set_page_config(page_title="ML Model Trainer", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center;'>🤖 ML Model Trainer App</h1>
    <p style='text-align: center; color: gray;'>Train, evaluate and export ML models with ease</p>
""", unsafe_allow_html=True)

st.divider() #Add a horizontal line

#INTERMEDIATE FEATURE: Cache Functionsto avoid reloading everytime
@st.cache_data
def load_seaborn_dataset(name):
    return sns.load_dataset(name)

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

#Session state variables to keep values across interactions
if "model_fitted" not in st.session_state:
    st.session_state.model_fitted = False
    st.session_state.pipeline = None
    st.session_state.y_test = None
    st.session_state.y_pred = None

#--STEP 3-- Dataset selection section

#Expandable section to choose which datset or upload which and displaying the datset preview
with st.expander("📁 Dataset selection", expanded=True):
    dataset_source = st.radio("Select dataset source:", ["Sample dataset", "Upload CSV"])
    if dataset_source == "Sample dataset":
        datasets = ["iris", "titanic", "tips"]
        dataset_name = st.selectbox("Choose a sample dataset", datasets)
        df = load_seaborn_dataset(dataset_name)
    else: #ADVANCED FEATURE- Custom dataset upload
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file to continue.")
            st.stop()
    st.success("✅ Preview of dataset:")
    st.dataframe(df, use_container_width=True)

st.divider()

#--STEP 4-- Feature and Target Selection section

#Section to choose input fetures and target variable
with st.expander("📌 Select features and target", expanded=True):
    #You automatically detect the numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("**🔢 Numerical columns detected:**") #Show the user teh columns that have been selected
    st.write(numerical_cols)
    st.markdown("**🔤 Categorical columns detected:**")
    st.write(categorical_cols)

    selected_features_num = st.multiselect("📊 Select numerical features", numerical_cols)
    selected_features_cat = st.multiselect("🔠 Select categorical features", categorical_cols)
    target_variable = st.selectbox("🎯 Select target variable (y)", df.columns)

selected_features = selected_features_num + selected_features_cat #Combine the numerical and categorical colums you have selected

#--STEP 5-- Model Configuration section
if selected_features:
    st.divider()
    with st.form("model_form"):
        st.subheader("🧪 Model Configuration")
        #INTERMEDIATE FEATURE ( Multiple model Implementation) Choose the model type you want
        model_type = st.selectbox("Select model", ["Linear Regression", "Random Forest Regressor",
                                                   "Logistic Regression", "Random Forest Classifier"])
        #Slider for the test size
        test_size = st.slider("Test size (%)", 10, 50, 20) / 100
        
        #Additional parameters depending on the model you chose
        if "Random Forest" in model_type:
            n_estimators = st.slider("Number of trees (n_estimators)", 10, 200, 100) #INTERMEDIATE FEATURE : Parameter Tunning Options ( Dynamic sliders and inputs per model)
            max_depth = st.slider("Max depth of trees", 1, 20, 5)
        else:
            n_estimators = None
            max_depth = None

        if model_type == "Logistic Regression":
            c_value = st.number_input("Inverse regularization strength (C)", min_value=0.01, max_value=10.0, value=1.0)
            max_iter = st.slider("Max iterations", 100, 1000, 500)
        else:
            c_value = None
            max_iter = None

        submit_button = st.form_submit_button("🚀 Fit model") #Fit model button for user

#--STEP 6-- Model Trainig section after the FIT 
    if submit_button:
        st.subheader("📡 Training in progress...")
        X = df[selected_features]
        y = df[target_variable]
        
        #Encode categorical target if needes
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        #Validation if classifier/ regression is possible 
        is_classifier = model_type in ["Logistic Regression", "Random Forest Classifier"]
        y_is_continuous = pd.api.types.is_float_dtype(y) and len(np.unique(y)) > 10

        if is_classifier and y_is_continuous:
            st.error("❌ You selected a classification model, but the target variable seems continuous.")
            st.stop()

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), selected_features_cat)],
            remainder="passthrough"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) #Split the data
        
        #Select model based on what the user put as input

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_type == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(C=c_value, max_iter=max_iter)

        pipeline = make_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        #Store the results in session state
        st.session_state.model_fitted = True
        st.session_state.pipeline = pipeline
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

#--STEP 7-- Results and Visualizations of the model chosen 
if st.session_state.model_fitted:
    st.divider()
    st.subheader("📊 Model Results")

    pipeline = st.session_state.pipeline
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    #Show performance metrics depending on model type chosen

    if "Regress" in model_type:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"🧮 Mean Squared Error (MSE): `{mse:.2f}`")
    else:
        acc = accuracy_score(y_test, y_pred)
        st.write(f"✅ Accuracy: `{acc:.2%}`")

    #Visualizations section
    st.divider()
    st.subheader("📈 Visualizations")

    #For regression : residual plot
    if "Regress" in model_type:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(residuals, kde=True, ax=ax, color="#8ECFC9") #INTERMEDIATE FEATURE VISUALIZATION: Residual Plot (regression),Confusion matrix ( classification),ROC Curve ( Binary Classification)
        ax.set_title("Residual Distribution", fontsize=14, weight="bold")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else: #For classificagtion confusion matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax, cbar=False)
        ax.set_title("Confusion Matrix", fontsize=14, weight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        #ROC curve only for binary classification
        if hasattr(pipeline.named_steps[model_type.lower().replace(' ', '')], "predict_proba") and len(np.unique(y_test)) == 2:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr, label="ROC curve", color="#ff7f0e")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve", fontsize=14, weight="bold")
            st.pyplot(fig)

    #Feature Importance only for Random Forest ( INTERMEDIATE FEATURE)
    if model_type in ["Random Forest Regressor", "Random Forest Classifier"]:
        importances = pipeline.named_steps[model_type.lower().replace(' ', '')].feature_importances_
        encoded = pipeline.named_steps["columntransformer"].named_transformers_["cat"].get_feature_names_out(selected_features_cat)
        feature_names = list(encoded) + selected_features_num
        series = pd.Series(importances, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        series.plot(kind="barh", ax=ax, color="#4c72b0")
        ax.set_title("Feature Importances", fontsize=14, weight="bold")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

#--STEP 8-- Download and export section - ADVANCED FEATURE

    st.divider()
    st.subheader("💾 Download Trained Model")
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    st.download_button("📥 Download model (.joblib)", buffer, file_name="trained_model.joblib")
