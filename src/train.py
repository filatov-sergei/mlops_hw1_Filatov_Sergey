import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

with open('params.yaml') as f:
    params = yaml.safe_load(f)['train']

train = pd.read_csv('data/processed/train.csv')

X = train.drop('target', axis=1)
y = LabelEncoder().fit_transform(train['target'])

if params['model_type'] == 'random_forest':
    model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
elif params['model_type'] == 'logistic_regression':
    model = LogisticRegression(random_state=params['random_state'])
else:
    raise ValueError("Неизвестный тип модели")

with mlflow.start_run():
    mlflow.log_param("model_type", params['model_type'])
    mlflow.log_param("random_state", params['random_state'])
    if params['model_type'] == 'random_forest':
        mlflow.log_param("n_estimators", params['n_estimators'])
    
    model.fit(X, y)
    
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    mlflow.log_metric("train_accuracy", acc)
    
    mlflow.sklearn.log_model(model, "model")
    
    joblib.dump(model, 'model.pkl')
    print("Обучение модели завершено")