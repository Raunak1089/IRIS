# train_xgb.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import joblib
import shap
import json

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Optional: binary variant (uncomment to do setosa vs others)
#y_binary = (y == 0).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale features (not strictly necessary for tree models but nice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost classifier (multiclass)
num_class = len(np.unique(y_train))

# Simple XGBClassifier fit for new XGBoost
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_class,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# Fit without eval_metric / early stopping
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", acc)
print("Confusion matrix:\n", cm)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ROC AUC (one-vs-rest)
y_test_binarized = label_binarize(y_test, classes=np.arange(num_class))
y_proba = model.predict_proba(X_test_scaled)
try:
    roc_auc = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovr')
    print("ROC AUC (macro, OVR):", roc_auc)
except Exception as e:
    print("ROC AUC couldn't be computed:", e)

# Train final model on full dataset (optional)
X_full_scaled = scaler.fit_transform(X)
final_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_class,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
final_model.fit(X_full_scaled, y)

# SHAP explainer (TreeExplainer is fast for tree models)
explainer = shap.TreeExplainer(final_model)
# We'll cache a small background sample for the explainer (use a subset)
background = shap.sample(pd.DataFrame(X_full_scaled, columns=iris.feature_names), 50, random_state=42)
# Optionally precompute expected_value etc by calling explainer on background
explainer_expected = explainer.expected_value

# Save artifacts
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'target_names': iris.target_names.tolist(),
    'feature_names': iris.feature_names
}, 'model_artifacts.joblib')

# Save a small serialized explainer background so Flask can reconstruct TreeExplainer quickly
joblib.dump({'background': background}, 'explainer_background.joblib')

print("Saved model_artifacts.joblib and explainer_background.joblib")
