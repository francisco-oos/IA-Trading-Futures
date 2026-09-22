import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def main():
    # Cargar datos combinados y ordenar por tiempo
    df = pd.read_csv("data/ohlcv_combined.csv", parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)

    # ---------------------------------------
    # CREAR COLUMNA TARGET si no existe
    if 'target' not in df.columns:
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Eliminar filas con valores faltantes
    df.dropna(inplace=True)
    # ---------------------------------------

    # Definir variables predictoras y target
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'future_close', 'target']]
    X = df[feature_cols]
    y = df['target']

    # Setup TimeSeriesSplit para validación temporal
    tscv = TimeSeriesSplit(n_splits=5)

    # Modelo base XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Parámetros para RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # Randomized Search con validación temporal
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=30,
        scoring='accuracy',
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("Iniciando búsqueda de hiperparámetros...")
    search.fit(X, y)

    print("\nMejores parámetros encontrados:")
    print(search.best_params_)

    # Entrenar modelo final con mejores parámetros usando todo el dataset
    best_xgb = search.best_estimator_
    best_xgb.fit(X, y)

    # Guardar modelo
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(best_xgb, "models/xgb_model_robust.joblib")
    print("\nModelo robusto guardado en models/xgb_model_robust.joblib")

    # Evaluar modelo usando TimeSeriesSplit y mostrar métricas
    print("\nEvaluación con validación temporal:")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_xgb.fit(X_train, y_train)
        preds = best_xgb.predict(X_test)

        print(f"\nFold {fold}:")
        print(classification_report(y_test, preds))
        print("Matriz de confusión:\n", confusion_matrix(y_test, preds))
        print("Accuracy:", accuracy_score(y_test, preds))

if __name__ == "__main__":
    main()
