from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
model = pickle.load(open(MODEL_PATH, "rb"))

EXPECTED_FEATURES = list(model.feature_names_in_)

# Reasonable defaults so frontend can provide a minimal set of fields.
NUMERIC_DEFAULTS = {
    "MSSubClass": 50.0,
    "LotArea": 9600.0,
    "OverallCond": 5.0,
    "YearBuilt": 1973.0,
    "YearRemodAdd": 1994.0,
    "BsmtFinSF2": 0.0,
    "TotalBsmtSF": 1080.0,
}


def _to_float(name: str, fallback: float) -> float:
    raw = request.form.get(name, "").strip()
    if raw == "":
        return fallback
    return float(raw)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html", prediction_text="Enter details and click Predict Price.")

    try:
        row = {feature: 0.0 for feature in EXPECTED_FEATURES}

        for feature, default in NUMERIC_DEFAULTS.items():
            row[feature] = _to_float(feature, default)

        row["HouseAge"] = row["YearRemodAdd"] - row["YearBuilt"]

        input_df = pd.DataFrame([row], columns=EXPECTED_FEATURES)
        prediction = model.predict(input_df)[0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction:,.2f}")
    except Exception as exc:
        return render_template("index.html", prediction_text=f"Prediction error: {exc}")


if __name__ == "__main__":
    app.run(debug=True)
