#LIBRARIES
import time as tm 
import numpy as np 
import math 
import random
import pandas as pd
import seaborn as sns
import os
import shutil
import kagglehub
from kagglehub import KaggleDatasetAdapter
import xgboost as xgb
from sklearn.metrics import accuracy_score


file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "blastchar/telco-customer-churn",
    file_path
)

print("First 5 records:")


df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.replace({'No': 0, 'Yes': 1})
df = df.select_dtypes(include='number')

print(df.head())

# Suppose you want the first 80% of data for training, last 20% for testing
train_size = int(0.8 * len(df))
val_size   = int(0.1 * len(df))

Y = df['Churn']
X = df.iloc[:, :-1]

x_train, y_train = X[:train_size], Y[:train_size] # 70% of the data
x_val, y_val     = X[train_size:val_size], Y[train_size:val_size] # 15% of the data (Between 0.7 and 0.85)
x_test, y_test   = X[val_size:], Y[val_size:] # 15% of the data (Between 0.85 and 1)

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------ 


clf_xgb = xgb.XGBClassifier(
    n_estimators=900,
    max_depth=9,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",
    eval_metric="auc",

    reg_alpha=0.0,        # L1 regularization
    reg_lambda=1.0,       # L2 regularization

    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),

    random_state=1,
    n_jobs=-1
)

clf_xgb.fit(x_train,y_train,verbose=True,eval_set=[(x_val,y_val)])


#Accuracy
y_pred = clf_xgb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#Assign to each customer churn prob
y_proba = clf_xgb.predict_proba(x_test)[:, 1]
x_test.assign(
    churn_probability=y_proba,
    actual_churn=y_test.values
).head()

