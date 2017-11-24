"""
    Functions in Matrix Factorization
"""
import numpy as np
import RecTool as rt
import bl_fun as bl
import random as rd


@rt.fn_timer
def mf_pq(para_m, para_k, epoch_n=10, learning_rate=0.005, reg=0.1):
    """
        Matrix factorization
        p[k_factors, users]
        q[k_factors, items]
        \hat{r}_{ij} = q_i^T p_u
        \min_{q,p} \sum{(r_{ij} - q_i^T p_u)^2 + \lambda (||q_i||^2 + ||p_u||^2)}
    :param para_m:          rating matrix
    :param para_k:          latent factor number
    :param epoch_n:         iteration times, default 20
    :param learning_rate:   learning rate, default 0.01
    :param reg:             regularization coefficient
    :return:                matrix p and q
    """
    num_user, num_item = para_m.shape

    # init with 1! Tt's silly to init with 0...
    p = np.matrix(np.ones([para_k, num_user]))
    q = np.matrix(np.ones([para_k, num_item]))

    # for f in range(n_factors):
    #       for _ in range(n_iter):
    #           for u, i, r in all_ratings:
    #               err = r_ui - <q[:f+1,i], p[:f+1, u]>
    #               update p[f, u]
    #               update q[f, i]
    for f in range(para_k):
        for i_epoch in range(epoch_n):
            for k, v in para_m.items():
                # k[0]: user
                # k[1]: item
                err = v - q[:f + 1, k[1]].T * p[:f + 1, k[0]]
                q[f, k[1]] += learning_rate * \
                    (err * p[f, k[0]] + reg * q[f, k[1]])
                p[f, k[0]] += learning_rate * \
                    (err * q[f, k[1]] + reg * q[f, k[0]])
    return p, q


@rt.fn_timer
def mf_with_bias(para_m, para_k, epoch_n=10, learning_rate=0.005, reg=0.1):
    """
        Matrix factorization with bias of users and items
        \hat{r}_{ij} = \mu + b_i + b_u + q_i^T p_u)^2
        \min_{q,p,b_i,b_u} \sum{(r_{ij} - \mu - b_i - b_u - q_i^T p_u)^2
                     + \lambda (||q_i||^2 + ||p_u||^2 + \sum{b_u}^2 + \sum{b_i}^2))
    :param para_m:          rating matrix
    :param para_k:          latent factor number
    :param epoch_n:         iteration times, default 20
    :param learning_rate:   learning rate, default 0.01
    :param reg:             regularization coefficient
    :return:                bu, bi and matrix p and q
    """
    num_user, num_item = para_m.shape
    mu = bl.get_mu(para_m)
    bu = np.matrix(np.zeros([num_user, 1]))
    bi = np.matrix(np.zeros([num_item, 1]))
    p = np.matrix(np.zeros([para_k, num_user])) + 0.1
    q = np.matrix(np.zeros([para_k, num_item])) + 0.1

    # for f in range(n_factors):
    #       for _ in range(n_iter):
    #           for u, i, r in all_ratings:
    #               err = r_ui - <q[:f+1,i], p[:f+1, u]>
    #               update p[f, u]
    #               update q[f, i]
    for f in range(para_k):
        for i_epoch in range(epoch_n):
            for k, v in para_m.items():
                # k[0]: user
                # k[1]: item
                err = v - mu - bu[k[0]] - bi[k[1]] - \
                    q[:f + 1, k[1]].T * p[:f + 1, k[0]]
                bu[k[0]] += learning_rate * (err - reg * bu[k[0]])
                bi[k[1]] += learning_rate * (err - reg * bi[k[1]])
                q[f, k[1]] += learning_rate * \
                    (err * p[f, k[0]] + reg * q[f, k[1]])
                p[f, k[0]] += learning_rate * \
                    (err * q[f, k[1]] + reg * q[f, k[0]])

    return mu, bu, bi, p, q


@rt.fn_timer
def predict(
        para_m,
        para_test,
        para_k,
        with_bias=False,
        epoch_n=10,
        learning_rate=0.05,
        reg=0.1):
    """
        Predict on the test data
    :param para_m:              rating matrix
    :param para_test:           test_data
    :param para_k:              latent factor number
    :param with_bias:           whether calculate with bias, default false
    :param epoch_n:             iteration times, default 20
    :param learning_rate:       learning rate, default 0.1
    :param reg:                 regularization coefficient
    :return :                   get the prediction score
    """
    res = []
    if with_bias:
        mu, bu, bi, p, q = mf_with_bias(
            para_m, para_k, epoch_n=epoch_n, learning_rate=learning_rate, reg=reg)
        for i in range(len(para_test)):
            k = para_test[i]
            hat = mu + bu[k[0]] + bi[k[1]] + q[:, k[1]].T * p[:, k[0]]
            res.append([para_test[i][0], para_test[i][1], hat])
    else:
        p, q = mf_pq(para_m, para_k, epoch_n=epoch_n,
                     learning_rate=learning_rate, reg=reg)
        for i in range(len(para_test)):
            k = para_test[i]
            hat = q[:, k[1]].T * p[:, k[0]]
            res.append([k[0], k[1], hat])
    return res


def predict_with_param(
        para_test,
        para_param,
        with_bias=False):
    res = []
    if with_bias:
        mu = para_param["mu"]
        bu = para_param["bu"]
        bi = para_param["bi"]
        p = para_param["p"]
        q = para_param["q"]
        for i in range(len(para_test)):
            k = para_test[i]
            hat = mu + bu[k[0]] + bi[k[1]] + q[:, k[1]].T * p[:, k[0]]
            res.append([para_test[i][0], para_test[i][1], hat])
    else:
        p = para_param["p"]
        q = para_param["q"]
        for i in range(len(para_test)):
            k = para_test[i]
            hat = q[:, k[1]].T * p[:, k[0]]
            res.append([k[0], k[1], hat])
    return res
