#Consider this function y=cos(x) and plot the function. Now from this function find
#discrete values of y for x values 0,1,2,3 and 4. Crate a table for the x and y values
#now using linear regression find values of y for appropriate x values. Plot for the
#regressed values and original function.
from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#for continuous function
x_con=np.linspace(0,4,100)
y_con=np.cos(x_con)


plt.plot(x_con,y_con,label="y=cos(x)",color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Discreate Value of y for x
x_dis=np.array([0,1,2,3,4]).reshape(-1,1)
y_dis=np.cos(x_dis)

#table create
print("x       y")
for i in range(len(x_dis)):
    print(f"{x_dis[i][0]}:{y_dis[i][0]}")
model=LinearRegression()
model.fit(x_dis,y_dis)

y_pred=model.predict(x_con.reshape(-1,1))
print(y_pred)

#Plot original function and regressed value

plt.plot(x_con,y_con,label="Original y=cos(x)", color='blue')
plt.scatter(x_dis,y_dis,color='red',label="Discrete points")
plt.plot(x_con,y_pred,label="Linear Regression fit",color='green')
plt.xlabel("x")
plt.ylabel("y")
plt.show()