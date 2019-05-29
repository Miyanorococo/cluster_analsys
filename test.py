from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

# sample choice
x_use = np.zeros(45)
x_test = np.zeros(5)
val = 10

for i in range(val):
    x_test = x[i::val]
    for j in range(5):
        if j == 0:
            x_use[0:i] = x[0:i]
        else:
            x_use[i+1:i+1+8] = x[i+2:i+2+8]
            x_use[i+1+10:i+1+19] = x[i+2+10:i+2+8+11]
            break


print(x)
print(x_use)
print(x_test)

# calculate design matrix
h = 0.1
k = calc_design_matrix(x, x, h)

# solve the least square problem
l = 0.3
theta = np.linalg.solve(
    k.T.dot(k) + l * np.identity(len(k)),
    k.T.dot(y[:, None]))
C = calc_design_matrix(x, x_test, h)
check = C.dot(theta)
error = np.mean((check.squeeze() - y_test) ** 2)

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(x, X, h)


# visualization
plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)
plt.savefig('lecture2-p44.png')
