#Python Implementation using linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("salary.csv")

print(df.head())

x=df[['years experience']]
y=df[['salary']]

#Scatter plot of data
plt.scatter(x,y,color='red')
plt.xlabel("Years experience")
plt.ylabel("Salary")
plt.title("Salary Data")
plt.show()

#Model Train using data set
model=LinearRegression()
model.fit(x,y)

x_value = np.array([[5]])  # Reshaping for sklearn
y_pred = model.predict(x_value)

print("Y prediction:",y_pred)

plt.plot(x,model.predict(x),color='blue')
plt.scatter(x,y,color='red')
plt.xlabel("Years experience")
plt.ylabel("Salary")
plt.title("Salary Data")
plt.show()

