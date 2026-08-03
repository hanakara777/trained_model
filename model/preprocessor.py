from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocessor(df):
    numerical_features = df.select_dtypes(['float64', 'int64']).columns.tolist()
    if 'death_event' in numerical_features:
        numerical_features.remove('death_event')

    ct_preprocessing = ColumnTransformer(
        [('numerical', StandardScaler(), numerical_features)]
    )
    
    return numerical_features, ct_preprocessing

__all__ = ['preprocessor']