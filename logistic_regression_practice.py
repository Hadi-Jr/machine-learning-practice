import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd

def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        p[i] = f_wb_i >= 0.5

    return p

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def cost_function(x, y, w, b):
    m, n = x.shape
    total_cost = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        total_cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    total_cost /= m
    return total_cost

def compute_gradient(X, y, w, b):
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += f_wb_i * X[i, j]
        dj_db += f_wb_i

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w_in, b_in)
            J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    return w_in, b_in, J_history, w_history

print("########################## DATA REPRESENTATION ##########################")
df = pd.read_csv("data/credit_score_numeric.csv")
x_train = df.drop(columns="credit_score").to_numpy()
y_train = df["credit_score"].to_numpy()
print(x_train)
print(y_train)

print("########################## INPUT NORMALIZATION ##########################")
mu = x_train.mean(axis=0)
sigma = x_train.std(axis=0)
x_norm = (x_train - mu) / sigma

print("########################## SIGMOID ##########################")
value = 0
print (f"sigmoid({value}) = {sigmoid(value)}")

print("########################## COST ##########################")
m, n = x_train.shape
initial_w = np.zeros(n)
initial_b = 0.

cost = cost_function(x_norm, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

print("########################## GRADIENT ##########################")
dj_db, dj_dw = compute_gradient(x_norm, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )

print("########################## GRADIENT DESCENT ##########################")
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(x_train.shape[1]) - 0.5)
initial_b = 0.
iterations = 1000
alpha = 0.1

w, b, J_history, w_history = gradient_descent(x_norm ,y_train, initial_w, initial_b,
                                   cost_function, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {w}, {b}")

fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(J_history)
ax2.plot(100 + np.arange(len(J_history[100:])), J_history[100:])
plt.show()

print("########################## TRAIN ACCURACY ##########################")
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3
tmp_X = np.random.randn(4, 2) - 0.5
tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')
p = predict(x_norm, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

