import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


def tuned_model_rfc(X_train, y_train, ct_processing):
    pipeline = Pipeline([
        ('preprocessor', ct_preprocessing),
        ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train=y_train.values.ravel()
    y_test=y_test.values.ravel()


    tuned_parameters = {
        'model__max_depth': [50],
        'model__n_estimators': [100],
        'model__min_samples_split': [4],
        'model__min_samples_leaf': [2],
        'model__max_features': ['sqrt']
    }

    tuned_model_rfc = RandomizedSearchCV(pipeline,tuned_parameters, cv=5, n_iter=50, error_score='raise' )
    tuned_model_rfc.fit(X_train, y_train)

    y_pred = tuned_model_rfc.predict(X_test)
    y_pred = y_pred.ravel()
    
    return tuned_model_rfc, pipeline

def save_rfc_model(model, filepath='models/saved_models/rfc_model.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_rfc_model(filepath='models/saved_models/rfc_model.pkl'):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

__all__ = ['tuned_model_rfc', 'save_rfc_model', 'load_rfc_model']