# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap

app = Flask(__name__)

# Load artifacts
artifacts = joblib.load('model_artifacts.joblib')
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']
target_names = artifacts['target_names']

# Load background for SHAP
exp_bg = joblib.load('explainer_background.joblib')['background']
# Build explainer
explainer = shap.TreeExplainer(model, data=exp_bg)

@app.route('/')
def index():
    return "Iris XGBoost prediction API. POST JSON to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON: {"instances": [[5.1, 3.5, 1.4, 0.2], [..], ...]}
    Returns: predicted class index, class name, probabilities
    """
    data = request.get_json(force=True)
    instances = data.get('instances')
    if instances is None:
        return jsonify({"error":"send JSON with 'instances' key"}), 400

    X = pd.DataFrame(instances, columns=feature_names)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled).tolist()
    preds = model.predict(X_scaled).tolist()
    response = []
    for i, instance in enumerate(instances):
        response.append({
            'input': instance,
            'predicted_index': int(preds[i]),
            'predicted_class': target_names[int(preds[i])],
            'probabilities': {target_names[j]: float(probs[i][j]) for j in range(len(target_names))}
        })
    return jsonify(response)

@app.route('/explain', methods=['POST'])
def explain():
    """
    Expects JSON: {"instances": [[...], ...], "top_k": 3}
    Returns SHAP values per instance (top_k features)
    """
    data = request.get_json(force=True)
    instances = data.get('instances')
    top_k = int(data.get('top_k', 3))
    if instances is None:
        return jsonify({"error":"send JSON with 'instances' key"}), 400

    X = pd.DataFrame(instances, columns=feature_names)
    X_scaled = scaler.transform(X)
    # SHAP values: for multiclass, shap_values is list of arrays per class
    shap_values = explainer.shap_values(X_scaled)  # list of arrays length = num_classes
    # For user-friendly explanation, compute for predicted class for each instance
    preds = model.predict(X_scaled)
    output = []
    for i, inst in enumerate(instances):
        pred_class = int(preds[i])
        # shap_value for predicted class
        sv = shap_values[pred_class][i]  # array length = n_features
        # pair features with absolute importance
        feats = list(zip(feature_names, sv))
        feats_sorted = sorted(feats, key=lambda x: abs(x[1]), reverse=True)
        top_feats = [{'feature': f, 'shap_value': float(v), 'abs': float(abs(v))} for f,v in feats_sorted[:top_k]]
        output.append({
            'input': inst,
            'predicted_class': target_names[pred_class],
            'top_features': top_feats
        })
    return jsonify(output)

if __name__ == '__main__':
    # For local testing only
    app.run(host='0.0.0.0', port=5000, debug=True)
