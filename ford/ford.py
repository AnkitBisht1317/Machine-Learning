import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as train_test_spilt
import sklearn.linear_model as LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
# Load DataSet
df=pd.read_csv('ford.csv')
df.head()
# Information about dataset
df.shape
df.describe()
df.info()
# Check Null value or not
df.isna().sum()
df.shape
df.drop_duplicates(inplace=True)
df.shape
sns.histplot(df['price'],bins = 50 , kde = True)
sns.heatmap(df.corr(numeric_only=True),annot=True)
sns.boxplot(data = df, x = 'year', y = 'price')
plt.xticks(rotation = 90)
sns.scatterplot(data = df, x = 'mileage', y = 'price')
sns.boxplot(data = df, x = 'engineSize', y = 'price')
sns.boxplot(data = df, x = 'transmission', y = 'price')
sns.boxplot(data = df, x = 'fuelType', y = 'price')
sns.boxplot(data = df, x = 'model', y = 'price')
plt.xticks(rotation = 90)
sns.boxplot(data = df, x = 'tax', y = 'price')
plt.xticks(rotation = 90)
X = df.drop('price', axis = 1)
y = df['price']
X_one_encode = pd.get_dummies(X, columns=['model','transmission','fuelType'],drop_first=True)
X_one_encode.astype(int)
X_one_encode.head
col = ['mileage',	'tax'	,'mpg']
scaler = StandardScaler()
X_one_encode[col] = scaler.fit_transform(X_one_encode[col])
X_one_encode.head()



Selected_feature = ['year', 'mileage', 'tax', 'mpg', 'engineSize',
    'model_ C-MAX','model_ EcoSport','model_ Edge','model_ Escort',
    'model_ Fiesta','model_ Focus','model_ Fusion','model_ Galaxy',
    'model_ Grand C-MAX','model_ Grand Tourneo Connect','model_ KA',
    'model_ Ka+','model_ Kuga','model_ Mondeo','model_ Mustang',
    'model_ Puma','model_ Ranger','model_ S-MAX','model_ Streetka',
    'model_ Tourneo Connect','model_ Tourneo Custom','model_ Transit Tourneo',
    'model_Focus','transmission_Manual','transmission_Semi-Auto',
    'fuelType_Electric','fuelType_Hybrid','fuelType_Other','fuelType_Petrol'
]

correlation = {
    feature: pearsonr(X_one_encode[feature], y)[0] 
    for feature in Selected_feature
}

# ✅ Convert dict items into DataFrame
correlation_df = pd.DataFrame(correlation.items(), columns=['features', 'pearson_correlation'])

# ✅ Sort by correlation value
correlation_df = correlation_df.sort_values(by='pearson_correlation', ascending=False)

print(correlation_df)
from scipy.stats import chi2_contingency
import pandas as pd

cat_feature = ['model_ C-MAX','model_ EcoSport','model_ Edge','model_ Escort',
    'model_ Fiesta','model_ Focus','model_ Fusion','model_ Galaxy',
    'model_ Grand C-MAX','model_ Grand Tourneo Connect','model_ KA',
    'model_ Ka+','model_ Kuga','model_ Mondeo','model_ Mustang',
    'model_ Puma','model_ Ranger','model_ S-MAX','model_ Streetka',
    'model_ Tourneo Connect','model_ Tourneo Custom','model_ Transit Tourneo',
    'model_Focus','transmission_Manual','transmission_Semi-Auto',
    'fuelType_Electric','fuelType_Hybrid','fuelType_Other','fuelType_Petrol']

alpha = 0.05   # usually 0.05, not 0.5
chi2_result = {}

for feature in cat_feature:
    contingency = pd.crosstab(X_one_encode[feature], y)  # ✅ y['price'] is 1D
    chi2_stats, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject NULL (keep feature)' if p_val < alpha else 'Accept NULL (remove feature)'
    chi2_result[feature] = {
        'chi2_stat': chi2_stats,
        'p_val': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_result).T
chi2_df = chi2_df.sort_values(by='p_val', ascending=True)
print(chi2_df)

fina_df = X_one_encode[['year','mileage','tax','mpg','engineSize','model_ Edge',
    'model_ Fiesta','model_ Fusion','model_ Galaxy','model_ KA',
    'model_ Ka+','model_ Kuga','model_ Mondeo','model_ Mustang',
    'model_ Puma','model_ S-MAX','model_ Streetka','model_ Tourneo Custom',
    'model_Focus','transmission_Manual','transmission_Semi-Auto',
    'fuelType_Electric','fuelType_Hybrid','fuelType_Petrol']]
fina_df

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Train-test split (fixed order of return values too)
X_train, X_test, y_train, y_test = train_test_split(fina_df, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# R² score
r2 = r2_score(y_test, y_pred)
print("R²:", r2)

# Adjusted R² score
n = X_test.shape[0]   # number of observations
p = X_test.shape[1]   # number of predictors
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R²:", adj_r2)







