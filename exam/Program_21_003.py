#Consider this function y=sin(x) and plot the function. Now from this function find
#discrete values of y for x values 0,1,2,3 and 4. Crate a table for the x and y values
#now using linear regression find values of y for appropriate x values. Plot for the
#regressed values and original function.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from pandas_for_beginers.Program_21_002 import y_pred

x_con=np.linspace(0,4,100)
y_con=np.sin(x_con)

plt.plot(x_con,y_con,label="continuous value",color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x_dis=np.array([0,1,2,3,4]).reshape(-1,1)
y_dis=np.sin(x_dis)

#create table
print("x    y")
for i in range(len(x_dis)):
    print(f"{x_dis[i][0]}: {y_dis[i][0]}")

#Create Model
model=LinearRegression()
model.fit(x_dis,y_dis)
y_pred=model.predict(x_con.reshape(-1,1))

#Plot function
plt.plot(x_con,y_con,label="continuous value",color='blue')
plt.scatter(x_dis,y_dis,label="discrete value",color='red')
plt.plot(x_con,y_pred,label="Regression Line",color='green')
plt.show()