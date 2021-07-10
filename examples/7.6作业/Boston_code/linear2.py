import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 30)

boston_dataset = load_boston()
# print(boston_dataset.keys())

boston = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)
# print(boston.head())

boston['MEDV'] = boston_dataset.target
# print(boston.head())

# print(boston.isnull().sum())

# Plot distribution of the target variable
sns.set(rc={
    'figure.figsize': (11.7, 8.27)
})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

# Correlation matrix
corr_matrix = boston.corr().round(2)
sns.heatmap(data=corr_matrix, annot=True)
plt.show()

# Observations
plt.scatter(boston['LSTAT'], boston['MEDV'], marker='o')
plt.title("LSTAT")
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

plt.scatter(boston['RM'], boston['MEDV'], marker='o')
plt.title("RM")
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()

# Applying Linear Regression
x = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=["LSTAT", "RM"])
y = boston['MEDV']

# Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(x_train, y_train)

# Model evaluation for training set
y_train_pred = model.predict(x_train)
mse = mean_squared_error(y_train, y_train_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_train_pred)
print("Model performance for training set.\n",
      "Root mean squared error: {}\n".format(rmse),
      "R2 Score: {}\n".format(r2))

# Model Evaluation for testing set
y_test_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)
print("Model performance for testing set.\n",
      "Root mean squared error: {}\n".format(rmse),
      "R2 Score: {}".format(r2))
