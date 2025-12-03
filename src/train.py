import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

with open('params.yaml') as f:
    params = yaml.safe_load(f)['train']

train = pd.read_csv('data/processed/train.csv')

X = train.drop('species', axis=1)
y = LabelEncoder().fit_transform(train['species'])

if params['model_type'] == 'random_forest':
    model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
elif params['model_type'] == 'logistic_regression':
    model = LogisticRegression(random_state=params['random_state'])
else:
    raise ValueError("Неизвестный тип модели")

model.fit(X, y)
joblib.dump(model, 'model.pkl')
print("Обучение модели завершено")