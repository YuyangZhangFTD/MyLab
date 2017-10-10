import random as rd
from scipy import sparse
from collections import defaultdict
import glmnet_py as glmnet
import RecTool as rt
import numpy as np


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
    rating_matrix = sparse.lil_matrix((max_user_num+1, max_item_num+1))
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
        glmnet_fit = glmnet.glmnet(x=A, y=aj, family='gaussian', alpha=alpha, nlambda=nlambda)
        new_wj = glmnet_fit["beta"][:, -1]
        for i in aj.indices:
            W[i, col] = new_wj[i]

    return W



@rt.fn_timer
def slim():
    pass


@rt.fn_timer
def _slim_coordinate_descent(
        X,
        y,
        alpha,
        beta,
        zero_position,
        max_iter,
        positive,
        tol=0.0001,
        random=True):
    """
             We minimize

            (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2

            subject to w[zero_position] = 0

                        w >= 0  (positive == True)

    """
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    norm_cols_X = np.sum(np.power(X, 2), axis=0)
    if alpha == 0 or beta == 0:
        print("Coordinate descent with no regularization may lead to"
              " unexpected results and is discouraged.")
        return None

    R = y - np.dot(X, w).reshape(n_samples,1)
    tol *= np.dot(y.T, y)

    for n_iter in range(max_iter):
        w_max = 0
        d_w_max = 0
        for f_iter in range(n_features):
            # keep constrain w[zero_position] = 0
            if f_iter == zero_position:
                continue

            # choose a dimension to optimize randomly or not
            if random:
                ii = np.random.randint(n_features)
            else:
                ii = f_iter

            # if columns of X are 0, w can't be updated
            if norm_cols_X[ii] == 0:
                continue

            # Store previous value
            w_ii = w[ii]

            if w_ii != 0:
                R += w_ii * X[:, ii].reshape(10, 1)

            tmp = np.sum(X[:, ii] * R)

            # keep positive for all w
            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = (np.sign(tmp) * np.max(np.abs(tmp) - alpha, 0)
                         / (norm_cols_X[ii] + beta))

            # update residual
            if w[ii] != 0.0:
                R -= w[ii] * X[:, ii].reshape(10, 1)


            # update the maximum absolute coefficient update
            d_w_ii = np.abs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if np.abs(w[ii]) > w_max:
                w_max = np.abs(w[ii])
        if w_max == 0.0 or d_w_max / w_max < tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller
            # than the tolerance: check the duality gap as ultimate
            # stopping criterion

            XtA = np.dot(X.T, R) - beta * w
            if positive:
                dual_norm_XtA = np.max(n_features, XtA)
            else:
                dual_norm_XtA = np.max(np.abs(n_features, XtA))

            R_norm2 = np.dot(R.T, R)

            w_norm2 = np.dot(w.T, w)

            if dual_norm_XtA > alpha:
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            l1_norm = np.sum(np.abs(w))

            gap += (alpha * l1_norm - const * np.dot(R.T, y) + 0.5
                    * beta * (1 + const ** 2) * (w_norm2))

            if gap < tol:
                # return if we reached desired tolerance
                break

    return w, gap, tol, n_iter + 1


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
    sample_num, feature_num = X.shape
    y = y.reshape([sample_num, 1])
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

            tmp = np.dot(X[:, feat_i], r)

            if positive and tmp < 0:
                w[feat_i] = 0
            else:
                w[feat_i] = np.sign(
                    tmp) * np.max(np.abs(tmp) - alpha, 0) / (norm_cols_X[feat_i] + beta)

            if w[feat_i] != 0:
                r -= w[feat_i] * X[:, feat_i].reshape([sample_num, 1])

            d_w_ii = np.abs(w[feat_i] - w_i)

            d_w_max = d_w_ii if d_w_ii > d_w_max else d_w_max

            w_max = np.abs(w[feat_i]) if np.abs(w[feat_i]) > w_max else w_max

        if w_max == 0.0 or d_w_max / w_max < tol or iter_i == iter_num - 1:
            break

    return w
