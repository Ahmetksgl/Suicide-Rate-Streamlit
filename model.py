import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
pd.set_option('display.max_columns', None)

df_ = pd.read_csv('master.csv')


print(df_.head(5))
print(df_.isnull().sum())


df = df_.drop(columns=['country-year', 'generation', 'HDI for year', ' gdp_for_year ($) '])
print(df.shape)
print(df.isnull().sum())


X = df.drop(columns=['suicides/100k pop', 'suicides_no'])  
y = df['suicides/100k pop'] 


X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"LightGBM - RMSE: {rmse:.2f}, R²: {r2:.2f}")


joblib.dump(model, 'lightgbm_model.pkl')
joblib.dump(X.columns, 'feature_names.pkl')
