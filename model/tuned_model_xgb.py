from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

def tuned_model_xbg(X_train, y_train, ct_preprocessing)
    pipeline_2 = Pipeline([
        ('preprocessor', ct_preprocessing),
        ('model', XGBClassifier(random_state=42))
    ])

    X_test, X_train, y_test, y_train = train_test_split(X,y, test_size=0.2, random_state=42)
    y_train=y_train.values.ravel()
    y_test=y_test.values.ravel()

    tuned_parameters_2 = {
        'model__max_depth':[25, 50, 75],
        'model__learning_rate':[0.5, 0.8, 0.9, 1],
        'model__n_estimators':[150, 165, 180, 190],
        'model__subsample':[0.7, 0.9, 1.0],
        'model__colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8]
    }

    tuned_model_xgb = RandomizedSearchCV(pipeline_2,tuned_parameters_2, cv=5, n_iter=50, error_score='raise' )
    tuned_model_xgb.fit(X_train, y_train)

    y_pred_2 = tuned_model_xgb.predict(X_test)
    y_pred_2 = y_pred_2.ravel()
    return tuned_model_xbg, tuned_parameters_2

def save_xgb_model(model, filepath='models/saved_models/rfc_model.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_rfc_model(filepath='models/saved_models/rfc_model.pkl'):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

__all__ = ['train_rfc', 'save_rfc_model', 'load_rfc_model']
