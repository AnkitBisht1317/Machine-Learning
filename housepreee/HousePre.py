import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df1 = pd.read_csv('house.csv')
df1.head()
df1.groupby('area_type')['area_type'].agg('count')
# remove extra Columns
df2 = df1.drop(['area_type','society','balcony','availability'], axis='columns')
df2.head()
# total number of null value
df2.isna().sum()
# fill null value
df2['location'].fillna(df2['location'].mode()[0], inplace=True)
df2['size'].fillna(df2['size'].mode()[0], inplace=True)
df2['bath'].fillna(df2['bath'].median(), inplace=True)
# feature engg
df2['size'].unique()
def extract_bhk(x):
    data = ""
    for i in x:
        if i == " ":
            break
        else:
            data = data + i
    return int(data)

df2['BHK'] = df2['size'].apply(extract_bhk)
df2['BHK'].unique()
df2['total_sqft'].unique()
def convert_sqft(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df2['total_sqft'] = df2['total_sqft'].apply(convert_sqft)
df2['price_per_sqft'] = df2['price']*100000/df2['total_sqft']
df2.head()
df2.location = df2.location.apply(lambda x : x.strip())
location_stats = df2.groupby('location')['location'].agg('count')
df2.location = df2.location.apply(lambda x : 'Other' if x in location_stats[location_stats <= 10].index else x)
df2.head()
# remove Outliers
df2 = df2[df2.total_sqft/df2.BHK >= 300] # which row remove which per room area is less than 300
df2.price_per_sqft.describe()
# drop those row which price_per_sqft area is moew than mena+std and less than mean-std

mean = df2.price_per_sqft.mean()
std = df2.price_per_sqft.std()

df2 = df2[(df2['price_per_sqft']>=(mean-std)) & (df2['price_per_sqft']<=(mean+std))]
df2 = df2[df2.bath<(df2.BHK+2)]
df2.drop(['size','price_per_sqft'], axis='columns', inplace=True)
dummies = pd.get_dummies(df2.location).astype(int)
df3 = pd.concat([df2, dummies.drop('Other',axis='columns')], axis='columns')
df3.drop('location', axis='columns', inplace=True)
X = df3.drop('price', axis='columns')
y = df3.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
# using cross fold validation
from sklearn.model_selection import cross_val_score
cross_val_score(LinearRegression(), X, y, cv=5)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


def find_best_model(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    result = []
    for name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=5, return_train_score=False)
        gs.fit(X, y)
        result.append({
            'model': name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(result, columns=['model', 'best_score', 'best_params'])
import pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))

!pip install streamlit pyngrok scikit-learn pandas numpy

from google.colab import files
uploaded = files.upload()
















