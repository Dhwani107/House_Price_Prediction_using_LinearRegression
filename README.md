# House Price Prediction (Linear Regression)

A simple ML + Flask project to predict house price from property details.

## Features

1. Data preprocessing (missing value handling + encoding)
2. Feature engineering (HouseAge)
3. Linear Regression model training
4. Model saved as model.pkl
5. Flask frontend for live prediction

## Why Results Are Weak

1. SalePrice missing rows are present in dataset
2. Notebook fills numeric columns with median, which can affect SalePrice target quality
3. Limited feature columns in dataset reduce model power
4. Linear Regression is too simple for this data and can underfit (miss non-linear relationships)
5. Single split evaluation (no cross-validation/tuning)

Current score is low due to both target handling issues and model underfitting.

## Shortcomings

1. No robust target filtering before training
2. Minimal model tuning
3. UI collects only a subset of raw inputs
4. No requirements.txt

## How To Run

1. Open terminal in this folder
2. Create venv:
   python -m venv .venv
3. Activate venv (PowerShell):
   .venv\Scripts\Activate.ps1
4. Install packages:
   pip install flask pandas numpy scikit-learn jupyter
5. Run app:
   python app.py
6. Open browser:
   http://127.0.0.1:5000

## Note

Use Flask URL (port 5000), not Live Server port 5500, for prediction.
