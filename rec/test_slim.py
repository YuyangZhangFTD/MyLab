import numpy as np
import slim_fun as s

X = np.random.random([10, 4])
w = np.array([1,2,0,3]).reshape(4,1)
y = np.dot(X, w)
res = s._slim_coordinate_descent(X,y,1,0.5,2,10,False)
print(res)
