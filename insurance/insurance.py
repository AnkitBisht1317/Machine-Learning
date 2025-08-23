import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# dataSets
df = pd.read_csv("insurance.csv")
# View Data
df.shape
df.info()
df.head()
df.describe()

# EDA
# missing value or not
df.isna().sum()
df.columns
# create Graph for numeric data
numeric_columns = ['age', 'bmi', 'children', 'charges']

# Loop through each numeric column and plot
for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue', edgecolor='black')


# Value Count
sns.countplot(x=df['sex'])
sns.countplot(x=df['smoker'])
sns.countplot(x=df['children'])
# Check Outliers we draw Boxplot
for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x = df[col])

# Check How to Column related to ech other
plt.figure(figsize = (8,6))
sns.heatmap(df.corr(numeric_only=True),annot = True)



            ## **************************************************Data Cleaning And Preprocessing*****************************************************************************


# remove Missing value and Duplicate value
df.isna().sum()
df_cleaned = df.copy() # create another copy of data (df_cleaned)

df_cleaned.shape # in this point shape is (1338,7)
df_cleaned.drop_duplicates(inplace=True)
df_cleaned.shape # check shape of data after and before remove duplicates(1337,7)
df_cleaned.head()

# convert Data type (object into int for numeric data)
df_cleaned.dtypes

# if our column is categorical then we perform encoding menas (0 or 1)
df_cleaned['sex'] = df_cleaned['sex'].map({"male" : 0, "female" : 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({"yes" : 1, "no" : 0})
df_cleaned.head()
# if our column is not categorical then we perform One-Hot Encode [new column create]
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
df_cleaned.head()
df_cleaned = df_cleaned.astype(int)
df_cleaned.head()
df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'], bins=[0,18.5,24.9,29.9,float('inf')], labels=['Underweight','Normal','Overweight','Obese'] )
df_cleaned.head()
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
df_cleaned = df_cleaned.astype(int)
df_cleaned.head()






# ******************************************************Feature Enggnering [all the distributed vlaue convert into +ve and -ve]********************************************************



cols = ['age','bmi','children']

scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
df_cleaned.head()


# find correlation [ by the help we find which column connect output column]


selected_features = ['age', 'bmi', 'children', 'sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest','bmi_category_Normal','bmi_category_Overweight','bmi_category_Obese']

correlation = {
    feature : pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}
correlation_df = pd.DataFrame(list(correlation.items()), columns=['feature', ' perarson correlation'])
correlation_df.sort_values(by=' perarson correlation', ascending=False)

# chi square test [find relation between output column to another column (it is for catagorical variable)]

cat_features = ['sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest','bmi_category_Normal','bmi_category_Overweight','bmi_category_Obese']

from scipy.stats import chi2_contingency

alpha = 0.05
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)

# Dictionary instead of list
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    decision = 'Reject Null {Keep Feature}' if p_val < alpha else 'Accept Null {Remove Feature}'

    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

# Convert to DataFrame outside loop
chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value', ascending=True)

print(chi2_df)

final_df = df_cleaned[['age','sex','bmi','children','smoker','charges','region_southeast','bmi_category_Obese']]
final_df



#*********************************************************************Linear Regression**********************************************************************************************


# Linear Regression Algorithm use
X = final_df.drop('charges', axis=1)
y = final_df['charges']

# devided train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

# perfrom linear regression to train model
model = LinearRegression()
model.fit(X_train, y_train)

# Preformance Matrices
y_pred = model.predict(X_test)

r2 = r2_score(y_pred,y_test)
print(r2)
# find Adjusted r2 score
n = X_test.shape[0] # only for row
p = X_test.shape[1] # only for column
adjusted_r2 = 1 - ((1-r2) * (n-1)/(n-p-1))
print(adjusted_r2)s