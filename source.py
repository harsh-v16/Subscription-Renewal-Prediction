# ==============================================================================
# PROJECT: SUBSCRIPTION RENEWAL PREDICTION
# DESCRIPTION: An end-to-end machine learning project to predict customer
#              subscription renewals. The primary business goal is to identify
#              customers at high risk of not renewing ('churning') so that the
#              company can proactively offer them incentives. This script covers
#              the entire workflow from data cleaning and EDA to model building,
#              evaluation, and generating a final, actionable list of at-risk
#              customers.
# ==============================================================================


# === Step 1: Import Essential Libraries ===
# Foundational libraries for data manipulation, analysis, and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn modules for preprocessing, model training, and evaluation.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler      # Scaler of choice for data with outliers.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Advanced modeling library for building a powerful predictive model.
from xgboost import XGBClassifier

# Advanced visualization library for creating interactive dashboards.
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# === Step 2: Load and Perform Data Quality Assurance ===
# Load the dataset and perform rigorous checks for data integrity.
print("Step 2: Loading and performing data quality checks...")
train_data = pd.read_csv('subscription_renewal_prediction.csv')

# --- Data Quality Check ---
# It's crucial to check for both standard nulls and hidden issues like blank spaces.
# Replace any cells containing only whitespace with NaN (Not a Number).
train_data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Now, check for any null values (including the newly replaced blank spaces).
if train_data.isnull().sum().sum() == 0:
    print("✅ Data Quality Check Passed: No null values or blank spaces found.")
else:
    print("⚠️ Warning: Null values detected. Further investigation and cleaning are needed.")
    print(train_data.isnull().sum())


# === Step 3: Exploratory Data Analysis (EDA) ===
# This phase is crucial for understanding the data's underlying patterns and
# identifying the relationships between features and the target variable ('renewed').
print("\nStep 3: Performing Exploratory Data Analysis (EDA)...")

# Visualize the class distribution to check for imbalance.
plt.figure(figsize=(8, 6))
sns.countplot(x='renewed', data=train_data)
plt.title('Distribution of Subscription Renewals')
plt.xlabel('Subscription Status (0 = Not Renewed, 1 = Renewed)')
plt.ylabel('Number of Users')
plt.show()

# Visualize how numerical features differ between renewing and non-renewing customers.
numerical_features = ['usage_days', 'last_login', 'monthly_fee']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_data, x=feature, hue='renewed', multiple='stack', bins=20, palette='viridis')
    plt.title(f'{feature.replace("_", " ").title()} Distribution by Renewal Status')
    plt.show()


# === Step 4: Data Preprocessing & Feature Engineering ===
# Prepare the data for modeling by separating features and the target, splitting
# the data, and scaling numerical features to prevent bias.
print("\nStep 4: Preprocessing data...")

# Define the features (X) and the target variable (y).
# 'user_id' is dropped as it is an identifier, not a predictive feature.
X = train_data.drop(columns=['user_id', 'renewed'])
y = train_data['renewed']

# Split the data into training (80%) and validation (20%) sets.
# `stratify=y` is essential here to ensure the proportion of renewals is the
# same in both the training and validation sets, which is critical for
# imbalanced datasets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features. RobustScaler is used because it is less sensitive
# to outliers than other scalers.
# IMPORTANT: We fit the scaler ONLY on the training data to prevent data leakage
# from the validation set. We then use the fitted scaler to transform both sets.
scaler = RobustScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])


# === Step 5: Build and Evaluate a Baseline Model ===
# We start with a simple Logistic Regression model. This helps us understand if
# the problem is simple and sets a performance benchmark that our more complex
# model must outperform.
print("\nStep 5: Training Baseline Logistic Regression Model...")

baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_val)

print("\n--- Baseline Model Classification Report ---")
print(classification_report(y_val, baseline_predictions, zero_division=0))

# --- Visualize Baseline Performance (with Error Handling) ---
# A `try...except` block is a professional programming practice. It prevents the
# script from crashing if the baseline model is too weak to predict both classes,
# which would otherwise cause a KeyError.
try:
    report_baseline = classification_report(y_val, baseline_predictions, output_dict=True)
    f1_baseline = report_baseline['weighted avg']['f1-score'] * 100

    fig_baseline = go.Figure(go.Indicator(
        mode="gauge+number",
        value=f1_baseline,
        title={'text': "<b>Baseline F1-Score</b>"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#DC3912"}}
    ))
    fig_baseline.update_layout(height=300)
    fig_baseline.write_html("baseline_performance_dashboard.html")

except KeyError:
    print("\n⚠️ Could not generate baseline dashboard because the model only predicted one class.")
    print("   This confirms the problem is non-trivial and requires a more advanced model.")


# === Step 6: Build the Optimized XGBoost Model ===
# The business goal is to find customers who will NOT renew (class 0).
# XGBoost is a powerful model, and we use `scale_pos_weight` to tell it to
# pay special attention to the minority class (the non-renewers),
# which directly aligns with our goal.
print("\nStep 6: Training Optimized XGBoost Model...")

# Calculate the weight to handle class imbalance.
scale_pos_weight = y_train.value_counts()[1] / y_train.value_counts()[0]

# Initialize and train the powerful XGBoost model.
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                          scale_pos_weight=scale_pos_weight, use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_val)

print("\n--- Optimized XGBoost Model Classification Report ---")
print(classification_report(y_val, xgb_predictions))


# === Step 7: Create XGBoost Performance Dashboard ===
# This dashboard visualizes the key performance metrics for our final,
# powerful model, focusing on the metrics for the at-risk customers (class 0).
print("\nStep 7: Creating XGBoost performance dashboard...")
report_xgb = classification_report(y_val, xgb_predictions, output_dict=True)

# Extract metrics for the "Not Renewed" class (class '0'), which is our target.
recall_xgb = report_xgb['0']['recall'] * 100
precision_xgb = report_xgb['0']['precision'] * 100
f1_xgb = report_xgb['0']['f1-score'] * 100

# Create the 3-gauge figure.
fig_xgb = make_subplots(rows=1, cols=3, specs=[[{'type': 'indicator'}]*3],
                      subplot_titles=("<b>Recall (Class 0)</b>", "<b>Precision (Class 0)</b>", "<b>F1-Score (Class 0)</b>"))

fig_xgb.add_trace(go.Indicator(mode="gauge+number", value=recall_xgb,
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#007BFF"}}), row=1, col=1)
fig_xgb.add_trace(go.Indicator(mode="gauge+number", value=precision_xgb,
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#28A745"}}), row=1, col=2)
fig_xgb.add_trace(go.Indicator(mode="gauge+number", value=f1_xgb,
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FFC107"}}), row=1, col=3)

fig_xgb.update_layout(title_text="🚀 XGBoost: At-Risk Customer Identification Performance", title_x=0.5, height=400)
fig_xgb.write_html("xgb_performance_dashboard.html")
print("\n✅ XGBoost performance dashboard saved to 'xgb_performance_dashboard.html'.")


# === Step 8: Generate Final Actionable List for the Business ===
# This is the ultimate deliverable. We use our best model (XGBoost) to create a
# prioritized list of customers, ranked by their probability of not renewing.
# This allows the marketing team to focus their efforts effectively.
print("\nStep 8: Generating the final list of at-risk customers...")

# Predict the probability of NOT renewing (class 0).
churn_probabilities = xgb_model.predict_proba(X_val)[:, 0]

# Create a final DataFrame with user IDs and their churn probability.
final_submission = pd.DataFrame({
    'user_id': train_data.loc[X_val.index, 'user_id'],
    'churn_probability': churn_probabilities
})

# Sort the list to bring the highest-risk customers to the top.
final_submission = final_submission.sort_values(by='churn_probability', ascending=False)

# Save the final list to a CSV file for the business team.
final_submission.to_csv('at_risk_customers.csv', index=False)

print("\n✅ Success! Actionable list saved to 'at_risk_customers.csv'.")