import pandas as pd
import matplotlib.pyplot as plt
#Column names
column_names = ['Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight','Rings']
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
df = pd.read_csv(url,names=column_names)
#Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
df_encoded = pd.get_dummies(df,columns=['Sex'])
df_encoded
X = df_encoded.drop('Rings',axis=1)
y = df_encoded['Rings']
#Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
#Prediction
y_pred = reg.predict(X_test)
#Evaluate
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)
print("Root Mean Squared Error: ",round(np.sqrt(mse)))
# Plot True vs Predicted Rings
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Actual vs Predicted Rings")
plt.show()
#Predicting Age
#Age = Rings + 1.5
y_test_age = y_test + 1.5
y_age = y + 1.5
print("Actual Age: ",y_age, "\nPredicted Age: ",y_test_age)
