import pandas as pd
import numpy as np

# Load the dataset and display first few rows of the data set
df = pd.read_csv('Downloads/heart.csv')  # Load the dataset from a CSV file into a DataFrame
print("First few rows of the dataset:")
print(df.head())  # Display the first few rows of the dataset

# Summary statistics
summary_stats = df.describe()  # Get summary statistics of the dataset
print("\nSummary statistics:")
print(summary_stats)  # Print the summary statistics

# Check for missing values
missing_values = df.isnull().sum()  # Check for missing values in each column
print("\nMissing values in each column:")
print(missing_values)  # Print the number of missing values in each column

# Check for duplicate rows
heart_data_dup = df.duplicated().any()  # Check if there are any duplicate rows
duplicate_rows = df.duplicated()  # Identify duplicate rows
print(f"Number of duplicate rows: {duplicate_rows.sum()}")  # Print the number of duplicate rows

# Replace '?' with NaN and convert to numeric (if applicable)
df.replace('?', pd.NA, inplace=True)  # Replace '?' with NaN
df = df.apply(pd.to_numeric, errors='coerce')  # Convert columns to numeric, setting errors to NaN

# Drop rows with missing values
df.dropna(inplace=True)  # Drop rows with any missing values

# Remove outliers based on a z-score threshold (e.g., |z| > 3)
from scipy import stats
z_scores = stats.zscore(df.select_dtypes(include=[float, int]))  # Calculate z-scores for numeric columns
abs_z_scores = abs(z_scores)  # Get absolute values of z-scores
filtered_entries = (abs_z_scores < 3).all(axis=1)  # Identify entries with all z-scores less than 3
df = df[filtered_entries]  # Keep only the entries without outliers

# Convert categorical columns to category data type
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
for col in categorical_columns:
    df[col] = df[col].astype('category')  # Convert specified columns to categorical data type

# Verify the data
print(df.info())  # Print a summary of the DataFrame including data types and non-null counts
print(df.describe())  # Print summary statistics
print(df.head())  # Display the first few rows of the dataset

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # Initialize the StandardScaler
scaled_features = scaler.fit_transform(df.drop('target', axis=1))  # Scale the features, excluding the target
print(scaled_features)  # Print the scaled features

# Convert scaled features back to a dataframe
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])  # Create a new DataFrame with scaled features
df_scaled['target'] = df['target'].values  # Add the target column back to the DataFrame
print("\nFirst few rows of the scaled dataset:")
print(df_scaled.head())  # Print the first few rows of the scaled dataset

# Correlation Analysis
import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = df_scaled.corr()  # Calculate the correlation matrix
plt.figure(figsize=(13, 9))  # Set the figure size for the plot
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # Plot the heatmap of the correlation matrix
plt.title("Correlation Matrix")  # Set the title of the heatmap
plt.show()  # Display the heatmap

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)  # Initialize the SelectKBest with f_classif and k=10
selected_features = selector.fit_transform(df_scaled.drop('target', axis=1), df_scaled['target'])  # Select the best features
print(selected_features)  # Print the selected features

# Get selected feature names
selected_feature_names = df_scaled.drop('target', axis=1).columns[selector.get_support()]  # Get the names of the selected features
print("\nSelected features:")
print(selected_feature_names)  # Print the selected feature names

# Split the data
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(selected_features, df_scaled['target'], test_size=0.2, random_state=42)  # Split the data into training and testing sets

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logreg = LogisticRegression()  # Initialize the Logistic Regression model
logreg.fit(X_train, y_train)  # Train the model on the training data
y_pred_logreg = logreg.predict(X_test)  # Predict the target values for the test data
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)  # Calculate the accuracy of the model
logreg_conf_matrix = confusion_matrix(y_test, y_pred_logreg)  # Generate the confusion matrix
logreg_class_report = classification_report(y_test, y_pred_logreg)  # Generate the classification report
print("\nLogistic Regression Accuracy:", logreg_accuracy)  # Print the accuracy
print("\nLogistic Regression Confusion Matrix:")
print(logreg_conf_matrix)  # Print the confusion matrix
print("\nLogistic Regression Classification Report:")
print(logreg_class_report)  # Print the classification report

# Visualize Logistic Regression Confusion Matrix
sns.heatmap(logreg_conf_matrix, annot=True, fmt='d', cmap='Blues')  # Plot the confusion matrix as a heatmap
plt.title("Logistic Regression Confusion Matrix")  # Set the title of the heatmap
plt.xlabel("Predicted")  # Set the x-axis label
plt.ylabel("True")  # Set the y-axis label
plt.show()  # Display the heatmap

# Support Vector Machine (SVM)
from sklearn.svm import SVC
svm = SVC(kernel='linear')  # Initialize the SVM with a linear kernel
svm.fit(X_train, y_train)  # Train the SVM on the training data
y_pred_svm = svm.predict(X_test)  # Predict the target values for the test data
svm_accuracy = accuracy_score(y_test, y_pred_svm)  # Calculate the accuracy of the model
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)  # Generate the confusion matrix
svm_class_report = classification_report(y_test, y_pred_svm)  # Generate the classification report
print("\nSVM Accuracy:", svm_accuracy)  # Print the accuracy
print("\nSVM Confusion Matrix:")
print(svm_conf_matrix)  # Print the confusion matrix
print("\nSVM Classification Report:")
print(svm_class_report)  # Print the classification report

# Visualize SVM Confusion Matrix
sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Blues')  # Plot the confusion matrix as a heatmap
plt.title("SVM Confusion Matrix")  # Set the title of the heatmap
plt.xlabel("Predicted")  # Set the x-axis label
plt.ylabel("True")  # Set the y-axis label
plt.show()  # Display the heatmap

# Neural Networks
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.001, early_stopping=True, random_state=42)  # Initialize the Neural Network with one hidden layer of 100 nodes
nn.fit(X_train, y_train)  # Train the Neural Network on the training data
y_pred_nn = nn.predict(X_test)  # Predict the target values for the test data
nn_accuracy = accuracy_score(y_test, y_pred_nn)  # Calculate the accuracy of the model
nn_conf_matrix = confusion_matrix(y_test, y_pred_nn)  # Generate the confusion matrix
nn_class_report = classification_report(y_test, y_pred_nn)  # Generate the classification report
print("\nNeural Network Accuracy:", nn_accuracy)  # Print the accuracy
print("\nNeural Network Confusion Matrix:")
print(nn_conf_matrix)  # Print the confusion matrix
print("\nNeural Network Classification Report:")
print(nn_class_report)  # Print the classification report

# Visualize Neural Network Confusion Matrix
sns.heatmap(nn_conf_matrix, annot=True, fmt='d', cmap='Blues')  # Plot the confusion matrix as a heatmap
plt.title("Neural Network Confusion Matrix")  # Set the title of the heatmap
plt.xlabel("Predicted")  # Set the x-axis label
plt.ylabel("True")  # Set the y-axis label
plt.show()  # Display the heatmap

# Cross-Validation
logreg_cv_scores = cross_val_score(logreg, selected_features, df_scaled['target'], cv=5)  # Perform 5-fold cross-validation for Logistic Regression
svm_cv_scores = cross_val_score(svm, selected_features, df_scaled['target'], cv=5)  # Perform 5-fold cross-validation for SVM
nn_cv_scores = cross_val_score(nn, selected_features, df_scaled['target'], cv=5)  # Perform 5-fold cross-validation for the Neural Network
print("\nLogistic Regression Cross-Validation Score:", logreg_cv_scores.mean())  # Print the mean cross-validation score for Logistic Regression
print("SVM Cross-Validation Score:", svm_cv_scores.mean())  # Print the mean cross-validation score for SVM
print("Neural Network Cross-Validation Score:", nn_cv_scores.mean())  # Print the mean cross-validation score for the Neural Network

# Interpret Model Predictions
logreg_coefficients = pd.DataFrame(logreg.coef_.T, index=selected_feature_names, columns=['Coefficient'])  # Create a DataFrame for Logistic Regression coefficients
svm_coefficients = pd.DataFrame(svm.coef_.T, index=selected_feature_names, columns=['Coefficient'])  # Create a DataFrame for SVM coefficients

# Neural Network Coefficients (weights)
nn_coefficients = pd.DataFrame(nn.coefs_[0], index=selected_feature_names, columns=[f'Hidden_Layer_Node_{i+1}' for i in range(100)])  # Create a DataFrame for Neural Network weights
print("\nLogistic Regression Coefficients:")
print(logreg_coefficients)  # Print the Logistic Regression coefficients
print("\nSVM Coefficients:")
print(svm_coefficients)  # Print the SVM coefficients
print("\nNeural Network Coefficients:")
print(nn_coefficients)  # Print the Neural Network weights
