import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

output_path = 'data/raw/data.csv'
df.to_csv(output_path, index=False)
print(f"Датасет Iris сохранён в {output_path}")