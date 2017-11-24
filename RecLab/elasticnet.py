import numpy as np
from sklearn import preprocessing


class elastic_net():

    def __init__(
            self,
            l1_ratio=0.5,
            alpha=1,
            iter_num=100,
            eps=1e-3,
            positive=True):

        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.iter_num = iter_num
        self.eps = eps
        self.positive = positive
        self.w = None

    def fit(self, X, y):

        X = preprocessing.scale(X)
        sample_num, feature_num = X.shape
        y = y.reshape(sample_num, 1)

        w = np.zeros([feature_num, 1]) + 0.1

        norm_cols_X = np.sum(np.power(X, 2), axis=0)


        r = y - np.dot(X, w)

        for iter_i in range(self.iter_num):
            for j in range(feature_num):

                if norm_cols_X[j] < self.eps:
                    continue
                
                if w[j] < -1 * self.eps or w[j] > self.eps:
                    r_j = r + x[:, j] * w[j] 

                # z = \frac{1}{N}\sum^N_{i=1}x_{ij}(y_i-\hat{y}_i^{j}) =
                # \frac{1}{N}\sum^N_{i=1}x_{ij}r_i+w_j
                z = np.mean(np.dot(X.T, r)) + w[j]
                
                z_wj = np.sum(np.dot(X[:,j].T, r_j))

                # gamma = alpha * l1_ratio
                gamma = self.alpha * self.l1_ratio

                if self.positive and z_wj < 0:
                    w[j] = 0.0
                else:
                    print('iter')
                    z = z_wj + w[j]
                    # soft-threshold
                    if gamma < abs(z):
                        if np.sign(z) > 0:
                            numerator = z - gamma
                        else:
                            numerator = z + gamma
                    else:
                        numerator = 0

                    denominator = 1 + self.alpha * (1 - self.l1_ratio)

                    w[j] = numerator / denominator
                print(w[j])
                  
                # update residual
                r = r_j - x[:, j] * w[j]
        return w


if __name__ == '__main__':

    x = np.random.random([100, 15])
    w = np.zeros([15, 1])
    w[1] = 1
    w[4] = 1
    w[9] = 2
    w[10] = 4
    w[11] = 1
    w[14] = 5
    err = np.random.normal(0, 1, [100, 1])
    y = np.dot(x, w) + err

    model = elastic_net(iter_num=5, positive=False, eps=1e-5)
    hat_w = model.fit(x, y)
    print(w)
    print(hat_w)
