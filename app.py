from pathlib import Path

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mumbai_housing.csv"

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

CATEGORICAL_FEATURES = ["ocean_proximity"]
TARGET = "median_house_value"

try:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET])

    df = df[df[TARGET] > 0].copy()

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    if len(df) < 20:
        raise ValueError("Not enough valid dataset rows to train the models.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_leaf=3,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    trained_models = {}
    model_results = []

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        trained_models[name] = pipeline

        model_results.append({
            "name": name,
            "r2": round(float(r2_score(y_test, predictions)), 3),
            "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
            "mse": round(float(mse), 2),
            "rmse": round(float(np.sqrt(mse)), 2)
        })

    # Use Random Forest as the main prediction model.
    model = trained_models["Random Forest"]
    test_predictions = model.predict(X_test)

    # Dataset-derived chart data.
    chart_data = {
        "labels": list(range(1, min(81, len(y_test)) + 1)),
        "actual": [
            round(float(x), 2) for x in y_test.iloc[:80]
        ],
        "predicted": [
            round(float(x), 2) for x in test_predictions[:80]
        ],
        "categories": [
            str(x) for x in df["ocean_proximity"].value_counts().index
        ],
        "categoryPrices": [
            round(float(df.loc[
                df["ocean_proximity"] == x, TARGET
            ].mean()), 2)
            for x in df["ocean_proximity"].value_counts().index
        ],
        "income": [
            round(float(x), 2) for x in df["median_income"].iloc[:80]
        ],
        "houseValues": [
            round(float(x), 2) for x in df[TARGET].iloc[:80]
        ]
    }

    dataset_rows = len(df)
    average_price = float(df[TARGET].mean())
    best_r2 = r2_score(y_test, test_predictions)
    main_mae = mean_absolute_error(y_test, test_predictions)

    startup_error = None

except Exception as exc:
    df = None
    model = None
    trained_models = {}
    model_results = []
    chart_data = {}
    dataset_rows = 0
    average_price = 0
    best_r2 = 0
    main_mae = 0
    startup_error = str(exc)


@app.route("/")
def home():
    if df is None:
        return render_template(
            "setup.html",
            error=startup_error
        ), 500

    categories = sorted(
        df["ocean_proximity"].dropna().astype(str).unique().tolist()
    )

    return render_template(
        "index.html",
        rows=dataset_rows,
        average_price=average_price,
        r2=float(best_r2),
        mae=float(main_mae),
        results=model_results,
        chart_data=chart_data,
        categories=categories
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not ready. Check the CSV file."}), 500

    try:
        data = request.get_json(silent=True) or {}

        longitude = float(data["longitude"])
        latitude = float(data["latitude"])
        age = float(data["age"])
        rooms = float(data["rooms"])
        bedrooms = float(data["bedrooms"])
        population = float(data["population"])
        households = float(data["households"])
        income = float(data["income"])
        proximity = str(data["proximity"])

        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if not 0 <= age <= 200:
            raise ValueError("Housing age must be between 0 and 200.")
        if not 1 <= rooms <= 100000:
            raise ValueError("Total rooms must be between 1 and 100000.")
        if not 1 <= bedrooms <= 50000:
            raise ValueError("Total bedrooms must be between 1 and 50000.")
        if not 1 <= population <= 1000000:
            raise ValueError("Population must be between 1 and 1000000.")
        if not 1 <= households <= 100000:
            raise ValueError("Households must be between 1 and 100000.")
        if not 0 < income <= 1000:
            raise ValueError("Median income must be greater than 0 and at most 1000.")

        valid_categories = set(df["ocean_proximity"].astype(str))
        if proximity not in valid_categories:
            raise ValueError("Please select a valid category.")

        row = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": age,
            "total_rooms": rooms,
            "total_bedrooms": bedrooms,
            "population": population,
            "households": households,
            "median_income": income,
            "ocean_proximity": proximity
        }])

        prediction = float(model.predict(row)[0])

        return jsonify({
            "price": round(prediction, 2),
            "formatted": "₹" + format(round(prediction), ","),
            "model": "Random Forest"
        })

    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc) or "Enter valid values."}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the entered values."}), 500


if __name__ == "__main__":
    app.run(debug=True)