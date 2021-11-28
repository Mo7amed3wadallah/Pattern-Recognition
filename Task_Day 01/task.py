
#Exercise 1,2
import numpy as np
import matplotlib.pyplot as plt

from prml.preprocess import PolynomialFeature
from prml.linear import (
        LinearRegression,
        RidgeRegression,
        BayesianRegression
        )

np.random.seed(1234)
# %% DEFINE THE FUNCTIONS

def create_dummy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std,size=x.shape)
    return x, t


def data_pattern(x):
    return np.sin(2 * np.pi * x)

# %% GENERATE TRAINING AND TEST DATA

x_train, y_train = create_dummy_data(data_pattern,
                                     10, 0.25)

x_test = np.linspace(0, 1, 100)
y_test = data_pattern(x_test)
# %% PLOT THE DATA
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

# PLOT THE SIN FUNCTION (x_test and y_test)
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")
plt.legend()
plt.show()
# %% Fitting a model of degree 0
degree = 2
feature = PolynomialFeature(degree)
x_train_2 = feature.transform(x_train)
x_test_2 = feature.transform(x_test)

model = LinearRegression()
model.fit(x_train_2, y_train)

y_2 = model.predict(x_test_2)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_2, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()

# %% Fitting a model of degree 0
degree = 2
feature = PolynomialFeature(degree)
x_train_2 = feature.transform(x_train)
x_test_2 = feature.transform(x_test)

model = LinearRegression()
model.fit(x_train_2, y_train)

y_2 = model.predict(x_test_2)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_2, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()

# %% Fitting a model of degree 3
degree = 3
feature = PolynomialFeature(degree)
x_train_3 = feature.transform(x_train)
x_test_3 = feature.transform(x_test)

model = LinearRegression()
model.fit(x_train_3, y_train)

y_3 = model.predict(x_test_3)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_3, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()

# %% Fitting a model of degree 4
degree = 4
feature = PolynomialFeature(degree)
x_train_4 = feature.transform(x_train)
x_test_4 = feature.transform(x_test)

model = LinearRegression()
model.fit(x_train_4, y_train)

y_4 = model.predict(x_test_4)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_4, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()

#%% CREATE THE RMSE FUNCTION 
def rmse(predicted_val, true_val):
    return np.sqrt(
        np.mean(
            np.square(
                predicted_val - true_val
                )
            )
        )
            
#%% PRINT RMSE VALUES OF THE TWO MODELS
rmse_2 = rmse(y_2, y_test)
rmse_3 = rmse(y_3, y_test)
rmse_4 = rmse(y_4, y_test)
print("The RMSE of a model of degree 2 is", 
      rmse_2)
print("The RMSE of a model of degree 3 is", 
      rmse_3)
print("The RMSE of a model of degree 4 is", 
      rmse_4)
#%% RMSE OF 10 MODELS
train_errors = []
test_errors = []
for i in range(10):
    feature = PolynomialFeature(i)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y = model.predict(X_test)
    z = model.predict(X_train)
    
    rmse_train = rmse(z, y_train)
    rmse_test = rmse(y, y_test)

    train_errors.append(rmse_train)
    test_errors.append(rmse_test)
#%% PLOT train_errors and test_errors    
plt.plot(train_errors, 'o-', 
         mfc="none", mec="b", 
         ms=10, c="b", label="Training")
plt.plot(test_errors, 'o-', 
         mfc="none", mec="r", 
         ms=10, c="r", label="Test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.show()

