import random as rd
from scipy import sparse
from collections import defaultdict
# import glmnet_py as glmnet
import RecTool as rt
import numpy as np
from sklearn import preprocessing


@rt.fn_timer
def slim_data_read(para_file, para_split_precent=0.8):
    user_dict = defaultdict(list)
    item_dict = defaultdict(list)
    test_data = []
    with open(para_file) as f:
        split_num = len(f.readline().split(','))
        if split_num == 3:
            is_movielens = True
            start_num = 1
        else:
            is_movielens = False
            start_num = 0
        while True:
            line = f.readline()
            if len(line) < 3:
                break
            line_array = line.split(",")
            # for MovieLens
            # userid, itemid, rating, time\n
            if is_movielens:
                vec = [
                    int(line_array[i]) - start_num
                    if i < 2
                    else float(line_array[i])
                    for i in range(3)
                ]
            # for WSDN and other data set
            # userid, itemid, rating\n
            else:
                vec = [
                    int(line_array[i]) - start_num
                    if i < 2
                    else float(line_array[i][:-1])
                    for i in range(3)
                ]

            # test data
            if rd.random() > para_split_precent:
                test_data.append(vec)
            # train data
            else:
                user_dict[vec[0]].append((vec[1], vec[2]))
                item_dict[vec[1]].append((vec[0], vec[2]))
    print("finish reading")
    max_user_num = max(user_dict.keys())
    max_item_num = max(item_dict.keys())
    rating_matrix = sparse.lil_matrix((max_user_num + 1, max_item_num + 1))
    for user, user_entry in user_dict.items():
        for item, rating in user_entry:
            rating_matrix[user, item] = rating
    return sparse.csc_matrix(rating_matrix), test_data, user_dict, item_dict


@rt.fn_timer
def slim_glmnet(para_m, alpha=0.5, nlambda=10):
    user_num, item_num = para_m.shape
    W = sparse.csc_matrix((item_num, item_num))

    # re-define name
    A = para_m.astype("float64")
    del para_m

    # min \|a_j - A*w_j\|^2_2 +
    # \gamma*(\alpha*\|W\|+(1-\alpha)\frac{1}{2}\|W\|^2_2)
    for col in range(item_num):
        aj = np.array(A.getcol(col).todense())
        glmnet_fit = glmnet.glmnet(
            x=A,
            y=aj,
            family='gaussian',
            alpha=alpha,
            nlambda=nlambda)
        new_wj = glmnet_fit["beta"][:, -1]
        for i in aj.indices:
            W[i, col] = new_wj[i]

    return W


@rt.fn_timer
def glmnet_own(
        X,
        y,
        alpha,
        beta,
        zero_pos=None,
        iter_num=10,
        tol=0.001,
        positive=False):
    """ GlmNet by koude

    \min_{w} \frac{1}{2N} \sum^N_{i=1} (y_i-X_iw)^2 + \beta P_{\alpha}(w)
    \P_{\alpha}(w) = (1-\alpha)\frac{1}{2} \|w\|^2_2 + \alpha\|w\|_1
                   = \sum^P_{j=1} (\frac{1}{2}(1-\alpha)w_j^2 + \alpha w_j)

    :param X:
    :param y:
    :param alpha:
    :param beta:
    :param zero_pos:
    :param iter_num:
    :param tol:
    :param positive:
    :return:
    """
    sample_num, feature_num = X.shape
    y = preprocessing.scale(y.reshape([sample_num, 1]))
    X = preprocessing.scale(X, axis=0)
    w = np.zeros((feature_num, 1))
    norm_cols_X = np.sum(np.power(X, 2), axis=0)

    # residual
    r = y - np.dot(X, w)
    # tolerance
    tol *= np.dot(y.T, y)

    for iter_i in range(iter_num):
        w_max = 0
        d_w_max = 0
        for feat_i in range(feature_num):
            if (zero_pos is not None and feat_i ==
                    zero_pos) or norm_cols_X[feat_i] == 0:
                continue

            w_i = w[feat_i]

            if w_i != 0:
                r += w_i * X[:, feat_i].reshape([sample_num, 1])

            tmp = np.dot(X[:, feat_i], r) / sample_num + w_i

            if positive and tmp < 0:
                w[feat_i] = 0
            else:
                # w[feat_i] = np.sign(
                # tmp) * np.max(np.abs(tmp) - alpha, 0) / (norm_cols_X[feat_i]
                # + beta)
                w[feat_i] = np.sign(tmp) * np.max(np.abs(tmp),
                                                  beta * alpha) / (1 + beta * (1 - alpha))

            if w[feat_i] != 0:
                r -= w[feat_i] * X[:, feat_i].reshape([sample_num, 1])

            d_w_i = np.abs(w[feat_i] - w_i)

            d_w_max = d_w_i if d_w_i > d_w_max else d_w_max

            w_max = np.abs(w[feat_i]) if np.abs(w[feat_i]) > w_max else w_max

        if w_max == 0.0 or d_w_max / w_max < tol or iter_i == iter_num - 1:
            break

    return w


def l1_gd(w):
    feature_num, _ = w.shape
    gd = np.ones([feature_num, 1])
    for i in range(feature_num):
        if w[i] < 0:
            gd *= -1
    return gd
        


def slim_gd(
        X,
        y,
        alpha,
        beta,
        zero_pos=None,
        iter_num=10,
        tol=0.001,
        learning_rate=0.001,
        positive=False):
    
    sample_num, feature_num = X.shape
    w = np.zeros([feature_num, 1]) 
   
    all_X = X
    all_y = y 
 
    for iter_i in range(iter_num):
        
        index = np.random.choice(sample_num, feature_num)
        X = all_X[index]
        y = all_y[index]

        print("iteration: "+str(iter_i))
        hat_y = np.dot(X,w)
        residual = y - hat_y

        loss = 0.5 * np.dot(residual.T, residual) + beta * (1 - alpha) * 0.5 * np.dot(w.T, w) + beta * alpha * np.sum(np.abs(w))
        print("loss: "+str(loss))
        
        gd = np.dot(X.T, residual) + beta * (1 - alpha) * w + beta * alpha * l1_gd(w)

        w += gd * learning_rate
        print("weight: ")
        print(w)
    pass



# test part
import numpy as np

X = np.random.random([10, 4])
w = np.array([0, 1, 2, 3]).reshape(4, 1)
y = np.dot(X, w)
# res = glmnet_own(X, y, alpha=0.5, beta=0.001, iter_num=1000, positive=True)
slim_gd(X, y, alpha=0, beta=0.5, iter_num=100, learning_rate=0.1)

