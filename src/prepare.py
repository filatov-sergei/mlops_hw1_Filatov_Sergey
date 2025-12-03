import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

# Загрузка параметров
with open('params.yaml') as f:
    params = yaml.safe_load(f)['prepare']

# Загрузка данных (предполагаем Iris или подобный; адаптируйте столбцы)
data = pd.read_csv('data/raw/data.csv')
data = data.dropna()  # Базовая очистка: удаление пропусков

# Сплит
train, test = train_test_split(
    data,
    train_size=params['split_ratio'],
    random_state=params['random_state']
)

# Сохранение
os.makedirs('data/processed', exist_ok=True)
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)

print("Подготовка данных завершена")