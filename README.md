# Readme
End-to-end ML implementation featuring data preprocessing, predictive modeling with XGBoost, model evaluation, and business recommendations for reducing customer attrition.

# -Executive Summary:
A production-ready machine learning system that predicts customer churn with 85% accuracy using XGBoost. Built by a student data science collective to help subscription businesses reduce attrition and increase lifetime value.

Business Impact: Potential to reduce churn by 25-35% through targeted interventions.

# -Problem Statement:
"For subscription-based companies, a 5% reduction in churn can increase profits by 25-95%." - Bain & Company

Customer churn costs businesses billions annually. This project delivers:

=> Early warning system for at-risk customers

=> Root cause analysis of why customers leave

=> Actionable retention strategies based on ML insights

# -Technical Implementation:

1. Data Cleaning
- Handled missing values in TotalCharges (2.1% of data)
- Encoded 16 categorical variables using one-hot encoding
- Normalized numerical features (tenure, MonthlyCharges, TotalCharges)

2. Feature Engineering
- Created tenure segments: New (<12m), Medium (12-36m), Loyal (>36m)
- Calculated charge-to-tenure ratio
- Added interaction terms between service types

- Models Compared
Model	               Accuracy	   Precision	   Recall	   ROC-AUC	      Best For
XGBoost	             85.2%	     0.83	         0.79	     0.87	          Best overall performance
Random Forest	       82.1%	     0.80	         0.75	     0.84	          Feature importance analysis
Logistic Regression	 80.5%	     0.78	         0.72	     0.82	          Interpretability & baseline
Neural Network	     83.7%	     0.81	         0.77	     0.85	          Complex pattern detection

- Optimal XGBoost Configuration
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,  # Handles class imbalance
    early_stopping_rounds=20,
    eval_metric='logloss'

)

Key Advantages:

-Handles class imbalance
-Feature importance reveals business insights
-Fast prediction suitable for real-time applications
-Robust to overfitting with regularization parameters

