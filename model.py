import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
pd.set_option('display.max_columns', None)

df_ = pd.read_csv('master.csv')

print(df_.head(5))

print(df_.isnull().sum())

# country                   0
# year                      0
# sex                       0
# age                       0
# suicides_no               0
# population                0
# suicides/100k pop         0
# country-year              0
# HDI for year          19456
#  gdp_for_year ($)         0
# gdp_per_capita ($)        0
# generation                0


df = df_.drop(columns=['country-year', 'generation', 'HDI for year', ' gdp_for_year ($) '])
print(df.shape)
print(df.isnull().sum())
# (27820, 8)

# num_cols = [col for col in df.columns if (df[col].dtype in ["int64","float64"]) & (df[col].nunique()>10)]
# print(num_cols)
# # ['year', 'suicides_no', 'population', 'suicides/100k pop', 'gdp_per_capita ($)']

# cat_cols = [col for col in df.columns if df[col].nunique()<10]
# print(cat_cols)
# ['sex', 'age']

# # Outlier Control

# def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
#     quartile1 = dataframe[col_name].quantile(q1)
#     quartile3 = dataframe[col_name].quantile(q3)
#     interquantile_range = quartile3 - quartile1
#     up_limit = quartile3 + 1.5 * interquantile_range
#     low_limit = quartile1 - 1.5 * interquantile_range
#     return low_limit, up_limit


# def check_outlier(dataframe, col_name):
#     low_limit, up_limit = outlier_thresholds(dataframe, col_name)
#     if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
#         return True
#     else:
#         return False


# print(check_outlier(df, num_cols))
# # True

# def replace_with_thresholds(dataframe, variable):
#     low_limit, up_limit = outlier_thresholds(dataframe, variable)
#     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
#     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# for col in num_cols:
#     replace_with_thresholds(df, col)

# print(check_outlier(df, num_cols))
# # False

X = df.drop(columns=['suicides/100k pop', 'suicides_no'])  
y = df['suicides/100k pop'] 

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest - RMSE: {rmse:.2f}, R²: {r2:.2f}")

joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(X.columns, 'feature_names.pkl')