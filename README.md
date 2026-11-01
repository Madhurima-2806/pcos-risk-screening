# PCOS Risk Screening App

A small web app that gives a **preliminary, symptom-based PCOS risk screening**. A user enters basic details (age, weight, height, period length) and answers yes/no questions about common symptoms. The app replies with one of three levels: fewer, some, or multiple indicators.

> **This is not a medical diagnosis.** It cannot confirm or rule out PCOS. Please consult a qualified healthcare professional for medical advice.

## How it works

1. **Data:** Two public survey datasets were merged and cleaned (999 people, about 28% with PCOS).
2. **EDA and preprocessing:** Charts and statistics were used to understand the data, then the data was prepared (filling blanks, scaling numbers, train/test split, class weights).
3. **Model:** Five models were compared with cross-validation. A Logistic Regression model performed best.
4. **Backend:** A Flask app (`app.py`) loads the saved model and exposes `POST /predict`.
5. **Frontend:** A single page (`templates/index.html`) with a step-by-step check-in, result screen, symptom summary, and a wellness corner with learning links.

## Model performance (honest numbers)

| Metric | Average over repeated tests |
|---|---|
| Accuracy | about 80% |
| Recall (PCOS cases caught) | about 76% |
| Precision | about 61% |
| ROC AUC | about 0.85 |

Symptoms alone have limits: many signs also occur in people without PCOS. The model is a screening aid only.

## Project structure

```
app.py                 Flask backend
requirements.txt       Python libraries
model/                 Saved model files (.joblib)
templates/index.html   Web page
static/rosie.png       Image used on the page
```

## Run it on your computer

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Deploy on Render

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Set the environment variable `PYTHON_VERSION` to the same Python version used to train the model.

## API

`POST /predict` with JSON such as:

```json
{"age": 24, "weight_kg": 60, "height_cm": 160, "period_length_days": 5,
 "regular_periods": 1, "weight_gain": 0, "excess_hair_growth": 1,
 "skin_darkening": 0, "hair_loss": 1, "pimples": 1,
 "fast_food": 0, "regular_exercise": 1}
```

Yes/no answers are 1 or 0 (for `regular_periods`, 1 means regular). The reply contains a risk score, a level (Low, Moderate, or High), and a disclaimer.

## Limitations

- Labels in the training data are mostly self-reported diagnoses, not lab results.
- The two source datasets differ (about 22% and 33% PCOS), so results may change on other groups.
- The sample is small (about 1,000 people) and mostly young adults.
- The risk score is not a true probability.

## Course project

Built as a course project for AI Applications Lab (data analysis and model training in Google Colab, web app deployed on Render).
