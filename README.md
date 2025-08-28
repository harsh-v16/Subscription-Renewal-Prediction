# 📌 Subscription Renewal Prediction

## 📖 Project Description

This repository contains a machine learning project designed to predict whether a customer will renew their subscription.

The primary business objective is to identify customers who are at high risk of not renewing, so the company can proactively offer them retention incentives and reduce churn.

The project compares a baseline Logistic Regression model with an optimized XGBoost model. While Logistic Regression provides a simple benchmark, the XGBoost model is selected as the final model due to its stronger ability to identify at-risk customers.

-----------

## 🚀 Features
	•	Data quality checks and preprocessing
	•	Exploratory Data Analysis (EDA) with visualizations
	•	Baseline model development using Logistic Regression
	•	Advanced model development using XGBoost with class imbalance handling
	•	Model evaluation using Recall, Precision, and F1-score (focused on non-renewals)
	•	Interactive HTML dashboards for model performance
	•	Final actionable list of high-risk customers ranked by churn probability

 --------

 ## 📂 Project Structure
 │── source.py                        # Main project script  
 
 │── baseline_performance_dashboard.html   # Baseline model performance dashboard  
 
 │── xgb_performance_dashboard.html       # XGBoost model performance dashboard  
 
 │── at_risk_customers.csv                # Final actionable list of high-risk customers

-------------

## ⚙ Installation & Usage
	1.	Clone this repository: git clone https://github.com/harsh-v16/Subscription-Renewal-Prediction.git cd Subscription-Renewal-Prediction

  2.	Install dependencies: pip install -r requirements.txt
    
  4. Run the project: python source.py

------------

## 📊 Outputs
	•	Baseline Performance Dashboard → baseline_performance_dashboard.html
	•	XGBoost Performance Dashboard → xgb_performance_dashboard.html
	•	Final Actionable List → at_risk_customers.csv (customers ranked by churn probability)

----------

## ✍ Author

Harsh Chaudhary
