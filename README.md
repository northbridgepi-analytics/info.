# Readme
End-to-end ML implementation featuring data preprocessing, predictive modeling with XGBoost, model evaluation, and business recommendations for reducing customer attrition.

# Executive Summary
A production-ready machine learning system that predicts customer churn with 85% accuracy using XGBoost. Built by a student data science collective to help subscription businesses reduce attrition and increase lifetime value.

Business Impact: Potential to reduce churn by 25-35% through targeted interventions.

# Problem Statement
"For subscription-based companies, a 5% reduction in churn can increase profits by 25-95%." - Bain & Company

Customer churn costs businesses billions annually. This project delivers:

=> Early warning system for at-risk customers

=> Root cause analysis of why customers leave

=> Actionable retention strategies based on ML insights

# Technical Implementation

1. Data Cleaning
- Handled missing values in TotalCharges (2.1% of data)
- Encoded 16 categorical variables using one-hot encoding
- Normalized numerical features (tenure, MonthlyCharges, TotalCharges)

2. Feature Engineering
- Created tenure segments: New (<12m), Medium (12-36m), Loyal (>36m)
- Calculated charge-to-tenure ratio
- Added interaction terms between service types

## Models Compared

| Model                | Accuracy | Precision | Recall | ROC-AUC | Best For |
|----------------------|----------|-----------|--------|---------|----------|
| **XGBoost**          | 85.2%    | 0.83      | 0.79   | 0.87    | Best overall performance |
| **Random Forest**    | 82.1%    | 0.80      | 0.75   | 0.84    | Feature importance analysis |
| **Logistic Regression** | 80.5% | 0.78      | 0.72   | 0.82    | Interpretability & baseline |
| **Neural Network**   | 83.7%    | 0.81      | 0.77   | 0.85    | Complex pattern detection |


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

# Business Insights & Recommendations

##Top 5 Churn Drivers Identified

Contract Type
- Month-to-month: 43% churn rate
- One-year: 11% churn rate
- Two-year: 3% churn rate
- Recommendation: Incentivize annual contracts with 10% discount

Tenure
- <12 months: 41% churn rate
- 12-36 months: 18% churn rate
- 36 months: 7% churn rate
- Recommendation: "First Year Success Program" for new customers

Payment Method 
- Electronic check: 45% churn rate
- Bank transfer: 9% churn rate
- Recommendation: Simplify electronic check process, promote auto-pay

Service Bundle 
- Fiber optic only: 42% churn rate
- Fiber + Premium services: 15% churn rate
- Recommendation: Bundle services to increase stickiness

Monthly Charges 
- $70: 38% churn rate
- <$70: 18% churn rate
- Recommendation: Tiered pricing with mid-range options

# Future Enhancements
 
Plan: Customer Acquisition & Sentiment Analysis
1. Social Media Sentiment → Predict brand perception
2. Website Analytics → Identify high-intent visitors
3. Market Basket Analysis → Cross-selling opportunities  
User sentiment → Feature extraction → XGBoost → Acquisition probability

Real-time Dashboard
- Live churn risk monitoring
- Automated retention email triggers
- A/B testing for intervention strategies

# Team
Northbridge Predictive Insightss
A group of mathematics and data science students building practical ML solutions for businesses.

Skills Demonstrated:  
-Data Science: XGBoost, Feature Engineering, Model Evaluation  
-Business Acumen: ROI Analysis, Retention Strategy, KPI Tracking  
-Engineering: Production Pipelines, API Development, Deployment  





# Contact
For Business Inquiries:
We offer free proof-of-concept churn analysis for qualified companies.
Email: northbridge.pi@gmail.com
GitHub: github.com/northbridgepi-analytics

