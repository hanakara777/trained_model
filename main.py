import os
from fastapi import FastAPI
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
from sklearn.metrics import classification_report, accuracy_score
from pydantic import BaseModel

os.makedirs('models', exist_ok=True)
model_path = 'models/rfc_model.pkl'

def train_and_save_model(save_path: str = model_path) -> None:
    print("Training new model...")

    records = fetch_ucirepo(id=519)
    X = records.data.features
    y = records.data.targets.values.ravel()

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(
            [("numerical", StandardScaler(), numerical_features)]
        )),
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=50,
            n_estimators=100,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Model saved to {save_path}")


if not os.path.exists(model_path):
    train_and_save_model()

model = joblib.load(model_path)

app = FastAPI(
    title="Heart Failure Prediction API",
    description=(
        "API for predicting heart failure using a machine learning model.\n\n"
        "Example payload:\n"
        "{\n"
        '"age": 65.0,\n'
        '"anaemia": 1,\n'
        '"creatinine_phosphokinase": 146.0,\n'
        '"diabetes": 0,\n'
        '"ejection_fraction": 20.0,\n'
        '"high_blood_pressure": 0,\n'
        '"platelets": 162000.0,\n'
        '"serum_creatinine": 1.3,\n'
        '"serum_sodium": 129.0,\n'
        '"sex": 1,\n'
        '"smoking": 1,\n'
        '"time": 7.0\n'
        "}"
    ),
    version="1.0",
)

class PatientData(BaseModel):
    age: float
    anaemia: int
    creatinine_phosphokinase: float
    diabetes: int
    ejection_fraction: float
    high_blood_pressure: int
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: int
    smoking: int
    time: float

@app.post("/predict")
async def predict_heart_failure(request: PatientData):
    input_data = pd.DataFrame([request.model_dump()])
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]
    classes = list(model.classses_)
    return { "prediction": int(prediction[0]),
        "prediction_label": "Death" if int(prediction[0]) == 1 else "Survive",
        "probability_survive": float(proba[classes.index(0)]),
        "probability_death": float(proba[classes.index(1)]),}

@app.get("/")
async def root():
    return {"message": "Welcome to the Heart Failure Prediction API. Use the /predict endpoint to make predictions."}
