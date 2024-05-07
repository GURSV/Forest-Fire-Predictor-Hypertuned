import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("/kaggle/input/ffpredict/Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
rand_f = RandomForestRegressor()
rand_f.fit(X_train, y_train)

# Grid Search CV for hypertuning.
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

# MAE for the regressor.
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Feature Importance plot.
feature_importance = best_rf.feature_importances_
print('Feature importance:', feature_importance)
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(10, 8))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(pd.DataFrame(data).columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Variable Importance')
plt.show()

pickle.dump(best_rf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))