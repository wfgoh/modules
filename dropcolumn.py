from sklearn.metrics import f1_score
from sklearn.base import clone

def dropcolumn(model, X_train, y_train, X_test, y_test, random_state=24):
    """
    This function calculate feature importance through dropping the feature
    Feature importance is set for weighted F1 score
    Return column names and feature importance
    """
    model_clone = clone(model)
    model_clone.random_state = random_state
    model_clone.fit(X_train, y_train)
    benchmark_score = f1_score(y_test, model_clone.predict(X_test),average='weighted')
    importances = []
    
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = f1_score(y_test, model_clone.predict(X_test.drop(col, axis = 1)),average='weighted')
        importances.append(benchmark_score - drop_col_score)

    return X_train.columns, importances
