import numpy as np

x = np.array([[1, 2], [1,4]])
y = np.array([[1, 4], [1,5]])
x_t = np.transpose(x)
inv = np.linalg.inv(x_t * x)
x_result = inv * x_t 
result = x_result * y
print(x_result)
print(result)
