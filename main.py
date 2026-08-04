import os
from fastapi import FastAPI, Request
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pydantic import BaseModel

os.makedirs('models', exist_ok=True)
model_path = 'models/rfc_model.pkl'
if not os.path.exists(model_path):
    print("Model file not found. New model will be trained and saved.")


    def heart_failure(data_set_id):
        heart_failure_clinical_records = fetch_ucirepo(id=519) 
        X = heart_failure_clinical_records.data.features 
        y = heart_failure_clinical_records.data.targets
        df = pd.concat([X,y], axis=1)
        return(df, X, y)

    df, X, y = heart_failure(519)

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'death_event' in numerical_features:
        numerical_features.remove('death_event')

    ct_preprocessing = ColumnTransformer(
        [('numerical', StandardScaler(), numerical_features)]
    )

    pipeline = Pipeline([
        ('preprocessor', ct_preprocessing),
        ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train=y_train.values.ravel()
    y_test=y_test.values.ravel()


    tuned_parameters = {
        'model__max_depth': [50],
        'model__n_estimators': [100],
        'model__min_samples_split': [4],
        'model__min_samples_leaf': [2],
        'model__max_features': ['sqrt']
    }

    tuned_model_rfc = GridSearchCV(pipeline,tuned_parameters, cv=5, error_score='raise' )
    tuned_model_rfc.fit(X_train, y_train)

    y_pred_1 = tuned_model_rfc.predict(X_test)
    y_pred_1 = y_pred_1.ravel()

    joblib.dump(tuned_model_rfc, 'rfc_model.pkl')

    rfc = RandomForestClassifier(random_state=42)

app = FastAPI(title="Heart Failure Prediction API", description="API for predicting heart failure using machine learning model.\n"
              " age: 65.0,\n"
              "anaemia: 1,\n"
              "creatinine_phosphokinase: 146.0,\n"
              "diabetes: 0,\n"
              "ejection_fraction: 20.0,\n"
              "high_blood_pressure: 0,\n"
              "platelets: 162000.0,\n"
              "serum_creatinine: 1.3,\n"
              "serum_sodium: 129.0,\n"
              "sex: 1,\n"
              "smoking: 1,\n"
              "time": 7.0", version="1.0")

model = joblib.load(model_path)

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
    probability = model.predict_proba(input_data)
    return {"prediction for death(0=Survive, 1=Death)": int(prediction[0]), "probability of survival": probability[0].tolist()}

@app.get("/")
async def root():
    return {"message": "Welcome to the Heart Failure Prediction API. Use the /predict endpoint to make predictions."}
