import numpy as np
import slim_fun as s


X = np.random.random([10, 4])
w = np.array([0,1,2,3]).reshape(4,1)
y = np.dot(X, w)
res = s.glmnet_own(X,y,alpha=0.5,beta=0.00001,iter_num=10)
print(res)
print(w)
