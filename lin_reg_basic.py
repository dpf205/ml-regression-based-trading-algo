import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# generate some test data
X = np.random.rand(100)
y = X + 0.1 * np.random.randn(100)

plt.scatter(X, y)
# plt.show()


# Step 1: Select type of model
from sklearn.linear_model import LinearRegression

# Step 2: select model hyperprameters
model = LinearRegression(fit_intercept=True)

# Step 3: Arrange data into a features matrix (independent X) and target vector (dependent y)
X = X.reshape(-1, 1)
print(X.shape)

# Step 4: Fit the model to your data
m_fit = model.fit(X, y)
print("\nmodel fit: ", m_fit)

m_coef = model.coef_
print("model coefficient: ", m_coef)

m_intercept = model.intercept_
print("model intercept: ", m_intercept)

# Step 5 Predict labels for unknown data
x_test = np.linspace(0, 1)
print("array of independent variables(randomly generated as an example): \n", x_test)

y_prediction = model.predict(x_test.reshape(-1,1))
plt.scatter(X,y)
plt.plot(x_test, y_prediction)
plt.show()
