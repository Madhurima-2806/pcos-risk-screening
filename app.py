"""
PCOS risk screening - backend (Flask)

What this file does:
  1. Loads the saved preprocessor and model once, when the app starts.
  2. Offers  POST /predict : receives a person's answers as JSON and returns a risk score.
  3. Offers  GET /health   : a quick check that the app is running.
  4. Shows templates/index.html at "/" once the frontend exists (next step).

Run it:   python app.py     then open  http://127.0.0.1:5000/health
"""

import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# __file__ does not exist when this code is run inside a notebook (Colab / Jupyter),
# so in that case we use the current working folder instead.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# The saved model files live in a folder called "model". If that folder is missing but the
# "processed" folder from the training notebook is here, use that one instead.
MODEL_DIR = os.path.join(BASE_DIR, "model")
if not os.path.isdir(MODEL_DIR):
    # no "model" folder: try "processed", otherwise the same folder as this file
    processed = os.path.join(BASE_DIR, "processed")
    MODEL_DIR = processed if os.path.isdir(processed) else BASE_DIR

# The web page lives in "templates" and the picture in "static". If those folders do not
# exist (all files uploaded side by side), look in the same folder as this file instead.
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(TEMPLATE_DIR):
    TEMPLATE_DIR = BASE_DIR
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = BASE_DIR

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR, static_url_path="/static")

# ------------------------------------------------------------------ load saved files
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
info = joblib.load(os.path.join(MODEL_DIR, "model_info.joblib"))

THRESHOLD = info.get("threshold", 0.5)
RAW_COLUMNS = list(preprocessor.feature_names_in_)    # the columns the preprocessor expects

# ------------------------------------------------------------------ what the form must send
# name: (minimum, maximum)    -> checked before we use the answer
NUMBER_FIELDS = {
    "age": (10, 60),
    "weight_kg": (30, 200),
    "height_cm": (120, 210),
    "period_length_days": (0, 20),
}
YES_NO_FIELDS = ["regular_periods", "weight_gain", "excess_hair_growth", "skin_darkening",
                 "hair_loss", "pimples", "fast_food", "regular_exercise"]

# Optional: if left out they are treated as "not answered" (filled in like the training data)
OPTIONAL_YES_NO = ["married", "pregnant"]
OPTIONAL_NUMBER = {"num_abortions": (0, 10)}


def clean_input(data):
    """Check the answers. Returns (clean_dict, list_of_error_messages)."""
    errors, clean = [], {}

    for name, (low, high) in NUMBER_FIELDS.items():
        try:
            value = float(data[name])
        except (KeyError, TypeError, ValueError):
            errors.append(f"'{name}' is missing or not a number")
            continue
        if not low <= value <= high:
            errors.append(f"'{name}' must be between {low} and {high}")
        clean[name] = value

    for name in YES_NO_FIELDS:
        if data.get(name) not in (0, 1, "0", "1"):
            errors.append(f"'{name}' must be 0 (no) or 1 (yes)")
        else:
            clean[name] = int(data[name])

    for name in OPTIONAL_YES_NO:
        if data.get(name) not in (None, "", 0, 1, "0", "1"):
            errors.append(f"'{name}' must be 0 or 1 if given")
        else:
            clean[name] = None if data.get(name) in (None, "") else int(data[name])

    for name, (low, high) in OPTIONAL_NUMBER.items():
        raw = data.get(name)
        if raw in (None, ""):
            clean[name] = None
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{name}' must be a number if given")
            continue
        if not low <= value <= high:
            errors.append(f"'{name}' must be between {low} and {high}")
        clean[name] = value

    return clean, errors


def predict_risk(clean):
    """Turn the cleaned answers into a risk score, using the SAME steps as in training."""
    row = dict(clean)
    row["bmi"] = row["weight_kg"] / (row["height_cm"] / 100) ** 2
    # In training, this flag was 1 for rows where the history questions were not asked
    row["history_missing"] = 1 if row.get("pregnant") is None else 0

    # one-row table with the columns in the exact order the preprocessor expects
    df = pd.DataFrame([row]).reindex(columns=RAW_COLUMNS)
    # blank answers must be real numbers (NaN), not text, so the preprocessor can fill them in
    optional = ["married", "pregnant", "num_abortions"]
    df = df.astype({c: float for c in optional if c in df.columns})
    processed = preprocessor.transform(df)

    # remove columns that were dropped before training (if any), keep the training order
    drop = [c for c in info.get("drop_after_preprocessing", []) if c in processed.columns]
    processed = processed.drop(columns=drop)[info["feature_columns"]]

    score = float(model.predict_proba(processed)[0, 1])
    return score, row["bmi"]


def risk_level(score):
    if score < 0.30:
        return "Low"
    if score < 0.60:
        return "Moderate"
    return "High"


# ------------------------------------------------------------------ pages and API
@app.route("/")
def home():
    if os.path.exists(os.path.join(TEMPLATE_DIR, "index.html")):
        return render_template("index.html")
    return jsonify({"message": "PCOS risk screening backend is running. The web form is the next step."})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": info.get("model_name"), "features": len(info["feature_columns"])})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send the answers as JSON."}), 400

    clean, errors = clean_input(data)
    if errors:
        return jsonify({"errors": errors}), 400

    score, bmi = predict_risk(clean)
    return jsonify({
        "risk_score": round(score, 3),
        "risk_percent": round(score * 100),
        "level": risk_level(score),
        "flagged": score >= THRESHOLD,
        "bmi": round(bmi, 1),
        "disclaimer": "This is a screening estimate based on symptoms, not a medical diagnosis. "
                      "Please consult a doctor for proper evaluation.",
    })


# Start the server only when this file is run on a computer ("python app.py").
# In Colab we do not start it (it cannot be opened from your browser there);
# use app.test_client() to try it instead.
if __name__ == "__main__" and "google.colab" not in sys.modules:
    app.run(debug=True)
