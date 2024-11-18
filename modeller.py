import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import joblib
from io import BytesIO
import matplotlib.pyplot as plt

# Function to load the dataset and extract features
@st.cache_data
def load_data(uploaded_file):
    dataframe = pd.read_csv(uploaded_file, header=0)

    # Check for missing values
    if dataframe.isnull().sum().any():
        st.warning("Warning: Missing values detected in the dataset. Consider handling them.")

    # Automatically set the last column as the target variable
    target_column = dataframe.columns[-1]
    feature_columns = dataframe.columns[:-1]

    X = dataframe[feature_columns]  # Features
    y = dataframe[target_column]     # Target

    return X, y, target_column

# Resampling Functions

# Split into Train and Test Sets
def split_train_test(X, Y, model_type):
    test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=7)
    
    model = LogisticRegression(max_iter=200) if model_type == 'Logistic Regression' else LinearRegression()
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)
    st.write("### Split into Train and Test Sets")
    st.write("Accuracy:", f"{accuracy * 100:.3f}%")

    model.fit(X, Y)  # Fit the model to the full dataset
    
    model_file = BytesIO()
    joblib.dump(model, model_file)
    model_file.seek(0)
    st.download_button("Download Model", model_file, file_name="split_train_test_model.joblib")

# K-Fold Cross-Validation
def kfold_cross_validation(X, Y, model_type):
    st.write("### K-Fold Cross Validation")
    num_folds = st.slider("Select number of folds:", 2, 10, 5)
    kfold = KFold(n_splits=num_folds)
    
    model = LogisticRegression(max_iter=210) if model_type == 'Logistic Regression' else LinearRegression()
    
    results = cross_val_score(model, X, Y, cv=kfold)
    st.write("Accuracy:", f"{results.mean() * 100:.3f}%")
    st.write("Standard Deviation:", f"{results.std() * 100:.3f}%")

    model.fit(X, Y)

    model_file = BytesIO()
    joblib.dump(model, model_file)
    model_file.seek(0)
    st.download_button("Download Model", model_file, file_name="kfold_cross_validation_model.joblib")

# Leave-One-Out Cross-Validation (LOOCV)
def loocv_cross_validation(X, Y, model_type):
    loocv = LeaveOneOut()
    model = LogisticRegression(max_iter=500) if model_type == 'Logistic Regression' else LinearRegression()
    
    results = cross_val_score(model, X, Y, cv=loocv)
    st.write("Accuracy:", f"{results.mean() * 100:.3f}%")
    st.write("Standard Deviation:", f"{results.std() * 100:.3f}%")

    model.fit(X, Y)

    model_file = BytesIO()
    joblib.dump(model, model_file)
    model_file.seek(0)
    st.download_button("Download Model", model_file, file_name="loocv_cross_validation_model.joblib")

# Repeated Random Test-Train Splits
def repeated_random_splits(X, Y, model_type):
    st.write("### Repeated Random Test-Train Splits")
    n_splits = st.slider("Select number of splits:", 2, 20, 10)
    test_size = st.slider("Select test size:", 0.1, 0.5, 0.33)
    seed = st.number_input("Random seed:", min_value=0, value=7)
    shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    
    model = LogisticRegression(max_iter=300) if model_type == 'Logistic Regression' else LinearRegression()
    
    results = cross_val_score(model, X, Y, cv=shuffle_split)
    st.write("Accuracy:", f"{results.mean() * 100:.3f}%")
    st.write("Standard Deviation:", f"{results.std() * 100:.3f}%")

    model.fit(X, Y)

    model_file = BytesIO()
    joblib.dump(model, model_file)
    model_file.seek(0)
    st.download_button("Download Model", model_file, file_name="repeated_random_splits_model.joblib")

# Performance Metric Functions
def classification_metrics(X, Y):
    st.sidebar.subheader("Select Classification Performance Metrics")
    selected_metrics = st.sidebar.multiselect(
        "Classification Metrics",
        ["Classification Accuracy (K-fold Cross Validation)",
                    "Classification Accuracy (split train-test 75:25 split ratio)",
                    "Logarithmic Loss",
                    "Confusion Matrix",
                    "Classification Report",
                    "Area Under ROC Curve"]
    )

    if "Classification Accuracy (K-fold Cross Validation)" in selected_metrics:
        num_folds = st.slider("Select number of folds for K-Fold Cross Validation:", 2, 20, 10)
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=None)
        model = LogisticRegression(max_iter=210)
        results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

        st.subheader("K-Fold Cross-Validation Results")
        st.write(f"Mean Accuracy: {results.mean() * 100:.3f}%")
        st.write(f"Standard Deviation: {results.std() * 100:.3f}%")
        plt.figure(figsize=(10, 5))
        plt.boxplot(results)
        plt.title('K-Fold Cross-Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks([1], [f'{num_folds}-Fold'])
        st.pyplot(plt)

    if "Classification Accuracy (split train-test 75:25 split ratio)" in selected_metrics:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)
        model = LogisticRegression(max_iter=210)
        model.fit(X_train, Y_train)
        accuracy = model.score(X_test, Y_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy * 100:.3f}%")

    if "Logarithmic Loss" in selected_metrics:
        num_folds = 10
        kfold = KFold(n_splits=num_folds, random_state=None)
        model = LogisticRegression(max_iter=200)
        scoring = 'neg_log_loss'
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

        log_loss_values = -results

        st.subheader("Cross-Validation Results")
        if np.any(np.isnan(log_loss_values)) or np.any(np.isinf(log_loss_values)):
            st.write("Warning: NaN or Inf values detected in LogLoss results.")
            log_loss_values = np.nan_to_num(log_loss_values, nan=0.0, posinf=0.0, neginf=0.0)

        st.write(f"Mean LogLoss: {log_loss_values.mean():.3f} (±{log_loss_values.std():.3f})")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_folds + 1), log_loss_values, marker='o', linestyle='-')
        plt.title('LogLoss for Each Fold of Cross-Validation')
        plt.xlabel('Fold Number')
        plt.ylabel('LogLoss')
        plt.xticks(range(1, num_folds + 1))
        plt.grid()

        y_max = log_loss_values.max() if not np.isnan(log_loss_values.max()) else 1.0  # Default to 1.0 if max is NaN
        plt.ylim(0, y_max + 0.5)
        st.pyplot(plt)

    if "Confusion Matrix" in selected_metrics:
        test_size = 0.33
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=7)
        model = LogisticRegression(max_iter=280)
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(Y_test, predicted)
        st.subheader("Confusion Matrix:")
        st.write(matrix)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=matrix).plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)


    if "Classification Report" in selected_metrics:
        test_size = 0.33
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=7)
        model = LogisticRegression(max_iter=180)
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        report = classification_report(Y_test, predicted, output_dict=True)
        st.subheader("Classification Report")
        st.text(classification_report(Y_test, predicted)) 
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)

    if "Area Under ROC Curve" in selected_metrics:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)
        model = LogisticRegression(max_iter=210)
        model.fit(X_train, Y_train)
        Y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
        st.subheader("ROC AUC")
        st.write("AUC: %.3f" % roc_auc)

        fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)


def regression_metrics(X, Y):
    st.sidebar.subheader("Select Regression Performance Metrics")
    selected_metrics = st.sidebar.multiselect(
        "Regression Metrics",
        ["Mean Squared Error (MSE)",
            "Mean Absolute Error (MAE)",
            "R-squared (R²)"
        ]
    )

    model = LinearRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    if "Mean Absolute Error (MAE)" in selected_metrics:
        kfold = KFold(n_splits=10, random_state=None)
        model = LinearRegression()
        scoring = 'neg_mean_absolute_error'
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        st.subheader("Cross-Validation Results")
        st.write(f"MAE: {-results.mean():.3f} (+/- {results.std():.3f})")

    if "Mean Squared Error (MSE)" in selected_metrics:
        test_size = 0.2
        seed = 42
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error (MSE): {mse:.3f}")

    if "R-squared (R²)" in selected_metrics:
        kfold = KFold(n_splits=10, random_state=None, shuffle=True)  # Use K-Fold for cross-validation

        # Train the data on a Linear Regression model
        model = LinearRegression()

        # Calculate the R² score using cross-validation
        scoring = 'r2'
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

        # Display results
        st.subheader("Model Evaluation")
        st.write(f"Average R² Score: {results.mean():.3f} (± {results.std():.3f})")

# Main App Function
def main():
    st.title("ML Resampling & Performance Metrics Tool")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        X, y, target_column = load_data(uploaded_file)
        st.write("### Dataset Preview")
        st.write(pd.DataFrame(X).head())

        # Sidebar option to select mode
        st.sidebar.header("Choose Mode")
        mode = st.sidebar.radio("Mode", ["Resampling Techniques", "Performance Metrics"])

        if mode == "Resampling Techniques":
            resampling = st.sidebar.selectbox("Select Resampling Technique", 
                                              ["Split into Train and Test Sets",
                                            "K-Fold Cross Validation",
                                            "Leave-One-Out Cross Validation",
                                            "Repeated Random Test-Train Splits"])
            model_type = st.sidebar.selectbox("Model Type", ["Logistic Regression", "Linear Regression"])

            if resampling == "Split into Train and Test Sets":
                split_train_test(X, y, model_type)
            elif resampling == "K-Fold Cross Validation":
                kfold_cross_validation(X, y, model_type)
            elif resampling == "Leave-One-Out Cross Validation":
                loocv_cross_validation(X, y, model_type)
            elif resampling == "Repeated Random Test-Train Splits":
                repeated_random_splits(X, y, model_type)

        elif mode == "Performance Metrics":
            metric_type = st.sidebar.radio("Metric Type", ["Classification", "Regression"])
            if metric_type == "Classification":
                classification_metrics(X, y)
            elif metric_type == "Regression":
                regression_metrics(X, y)

if __name__ == "__main__":
    main()
