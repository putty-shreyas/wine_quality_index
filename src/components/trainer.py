# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv(os.path.join("artifacts", "winequality.txt"), sep=";")

# Null values check
#print(dataset.isnull().sum())

# Dataset Splitting 
X = dataset.drop("quality", axis = 1)
y = dataset["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Feature Selection (Pearson Correlation)
plt.figure(figsize=(12,10))
X_corr = X_train.corr()
sns.heatmap(X_corr, cmap = "Blues", annot=True)
plt.title('Pearson Correlation of Features')
plt.savefig(os.path.join("artifacts", "corr_matrix.png"), dpi = 200)
plt.close()

# Feature Scaling
MMS = MinMaxScaler()
X_train_scaled = MMS.fit_transform(X_train)
X_test_scaled = MMS.transform(X_test)

# Model Training
regressor = RandomForestRegressor()
regressor.fit(X_train_scaled, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test_scaled)

'''
# Visualising the results
plt.scatter(X_test_scaled, y_test.reshape(X_test_scaled.shape), color='black', label='Actual')
plt.plot(X_test_scaled, y_pred, color='blue', linewidth=3, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression Results')
plt.legend()
plt.show()

#plt.figure(figsize=(10,8))
#plt.plot(y_pred, color ='red')
#plt.plot(y_test, color = 'blue')
#plt.title('Quality (Predicted)')
#plt.savefig(os.path.join("artifacts", "results.png"), dpi = 200)
'''
# Check Model performance

# (MSE)
print("MSE = ",mean_squared_error(y_test,y_pred))

# (RMSE)
print("RMSE = ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Checking Model performance (R2)
r2 = r2_score(y_test,y_pred)
print("R2 = ", r2)