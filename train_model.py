import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# LOAD DATASETS
# ==========================================

application_df = pd.read_csv(
    "dataset/application_record.csv"
)

credit_df = pd.read_csv(
    "dataset/credit_record.csv"
)

print("=" * 50)
print("APPLICATION DATASET")
print("=" * 50)

print(application_df.head())

print("\nShape:", application_df.shape)

print("\n" + "=" * 50)
print("CREDIT DATASET")
print("=" * 50)

print(credit_df.head())

print("\nShape:", credit_df.shape)


# ==========================================
# CREATE TARGET
# ==========================================

credit_df["TARGET"] = credit_df["STATUS"].apply(
    lambda x: 1
    if str(x) in ["1", "2", "3", "4", "5"]
    else 0
)

print("\nTARGET VALUE COUNTS")

print(
    credit_df["TARGET"].value_counts()
)


# ==========================================
# ONE TARGET PER CUSTOMER
# ==========================================

credit_target = (
    credit_df
    .groupby("ID")["TARGET"]
    .max()
    .reset_index()
)


# ==========================================
# MERGE DATASETS
# ==========================================

final_df = application_df.merge(
    credit_target,
    on="ID",
    how="inner"
)

print("\nMerged Dataset Shape:")

print(
    final_df.shape
)

print("\nFinal Target Counts:")

print(
    final_df["TARGET"].value_counts()
)


# ==========================================
# SELECT FEATURES
# ==========================================

features = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AMT_INCOME_TOTAL",
    "CNT_CHILDREN",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE"
]

X = final_df[features].copy()

y = final_df["TARGET"].copy()


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns


X[categorical_columns] = (
    X[categorical_columns]
    .fillna("Unknown")
)


X[numeric_columns] = (
    X[numeric_columns]
    .fillna(
        X[numeric_columns].median()
    )
)


# ==========================================
# ONE HOT ENCODING
# ==========================================

X = pd.get_dummies(
    X,
    drop_first=True
)

print("\nEncoded Feature Shape:")

print(
    X.shape
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Shape:")

print(
    X_train.shape
)

print("\nTesting Shape:")

print(
    X_test.shape
)


# ==========================================
# SMOTE
# ==========================================

smote = SMOTE(
    sampling_strategy=1.0,
    random_state=42
)


X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train,
        y_train
    )
)


print("\nSMOTE Target Counts:")

print(
    y_train_smote.value_counts()
)


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=15000
)

lr.fit(
    X_train_smote,
    y_train_smote
)


# ==========================================
# DECISION TREE
# ==========================================

print("Training Decision Tree...")

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train_smote,
    y_train_smote
)


# ==========================================
# RANDOM FOREST
# ==========================================

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=7,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=45,
    n_jobs=-1
)

rf.fit(
    X_train_smote,
    y_train_smote
)


# ==========================================
# XGBOOST
# ==========================================

print("Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb.fit(
    X_train_smote,
    y_train_smote
)


print("\nALL MODELS TRAINED SUCCESSFULLY!")


# ==========================================
# PREDICTIONS
# ==========================================

lr_pred = lr.predict(
    X_test
)

dt_pred = dt.predict(
    X_test
)

rf_pred = rf.predict(
    X_test
)

xgb_pred = xgb.predict(
    X_test
)


# ==========================================
# MODEL COMPARISON
# ==========================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)


models_predictions = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred
}


for model_name, prediction in models_predictions.items():

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    print(
        f"\n{model_name}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


# ==========================================
# RANDOM FOREST REPORT
# ==========================================

print("\n" + "=" * 50)
print("RANDOM FOREST REPORT")
print("=" * 50)


print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            "LOWER RISK",
            "HIGHER RISK"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


# ==========================================
# SAVE RANDOM FOREST MODEL
# ==========================================

joblib.dump(
    rf,
    "credit_card_model.pkl"
)


joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)


print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print(
    "Model saved as: "
    "credit_card_model.pkl"
)

print(
    "Feature columns saved as: "
    "feature_columns.pkl"
)