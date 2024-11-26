
# consider the function y=cos(x) and plot the function. Now from this function find discrete values of y for x values 0,1,2,3 and 4. Create a table for the x and y values. Now using linear regression find values of y  for appropriate x values. Plot for the regressed values and the original function using python
# Re-import libraries due to reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define the function y = cos(x)
def original_function(x):
    return np.cos(x)

# Generate x values and corresponding y values for the original function
x_continuous = np.linspace(0, 4, 500)  # Continuous x values for plotting
y_continuous = original_function(x_continuous)

# Discrete x values
x_discrete = np.array([0, 1, 2, 3, 4])
y_discrete = original_function(x_discrete)

# Create a table for discrete x and y values
table = np.column_stack((x_discrete, y_discrete))

# Perform linear regression
x_discrete_reshaped = x_discrete.reshape(-1, 1) # Reshape for sklearn

model = LinearRegression()
model.fit(x_discrete_reshaped, y_discrete)

# Predict y values for the original x values using the regression model
y_regressed = model.predict(x_discrete_reshaped)


# Generate regression line for plotting
x_regression = np.linspace(0, 4, 500).reshape(-1, 1)
y_regression = model.predict(x_regression)


# Plot the original function and regressed values
plt.figure(figsize=(10, 6))
plt.plot(x_continuous, y_continuous, label="Original Function (y = cos(x))", color="blue", linewidth=2)
plt.scatter(x_discrete, y_discrete, color="red", label="Discrete Points", zorder=5)
plt.plot(x_regression, y_regression, label="Linear Regression", color="green", linestyle="--", linewidth=2)
plt.title("Original Function and Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

