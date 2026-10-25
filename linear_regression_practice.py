import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy, math

def predict(x, w, b):
    return np.dot(x, w) + b

def compute_cost(x, y, w, b):
    m = x.shape[0]
    j_wb = 0.

    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        j_wb += (f_wb_i - y[i])**2
    j_wb /= (2 * m)
    return j_wb

def compute_gradient(x, y, w, b):
    m, n = x.shape

    dj_db = 0.
    dj_dw = np.zeros(n)

    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    dj_db /= m
    dj_dw /= m

    return dj_db, dj_dw

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    W_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            W_history.append(w)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history, W_history

print("########################## DATA REPRESENTATION ##########################")
df = pd.read_csv("data/house_price_regression_dataset.csv")
x_train = df[["Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]].to_numpy()
y_train = df["House_Price"].to_numpy()
print("First five elements of x_train are:", x_train[:5])
print("First five elements of y_train are:", y_train[:5])

print("########################## INPUT NORMALIZATION ##########################")
mu = x_train.mean(axis=0)
sigma = x_train.std(axis=0)
x_norm = (x_train - mu) / sigma

print("########################## DATA VISUALIZATION ##########################")
X_features = ["Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]
features_count = x_train.shape[1]
fig, ax = plt.subplots(1, features_count, figsize=(4 * features_count, 4))
for i in range(features_count):
    ax[i].scatter(x_norm[:, i], y_train, marker='x', c = 'r')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel('Price')

print("########################## COMPUTE COST ##########################")
w_example = [2, 1, 4, 2, 1, 1, 1]
b_example = 100
print(compute_cost(x_norm, y_train, w_example, b_example))

print("########################## COMPUTE GRADIENT ##########################")
dj_db_example, dj_dw_example = compute_gradient(x_norm, y_train, w_example, b_example)
print(dj_db_example)

print("########################## GRADIENT DESCENT ##########################")
initial_w = np.zeros_like(w_example)
initial_b = 0.
iterations = 4000
alpha = 0.01
w_final, b_final, J_hist, W_hist = gradient_descent(x_norm, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final}, {w_final}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])

print("########################## PREDICTION ##########################")
x_input = [1500, 5, 2, 2010, 2.6, 1, 8]
X_input_norm = (x_input - mu) / sigma
prediction = predict(X_input_norm, w_final, b_final)
print(f'The house costs ${prediction}')

# Display Cost / Ws
W_hist = np.array(W_hist)
fig, axes = plt.subplots(1, features_count, figsize=(4 * features_count, 4))
for j in range(features_count):
    axes[j].plot(W_hist[:, j], J_hist)
    axes[j].set_xlabel(f"w[{j}] ({X_features[j]})")
axes[0].set_ylabel("Cost J")
plt.show()