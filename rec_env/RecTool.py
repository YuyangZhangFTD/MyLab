"""
    Evaluation Functions in Collaborative Filter 
"""
import random as rd
import numpy as np
import pandas as pd
import time
from scipy import sparse
from functools import wraps


# record time
def fn_timer(function):
	@wraps(function)
	def function_timer(*args, **kwargs):
		start = time.clock()
		result = function(*args, **kwargs)
		end = time.clock()
		print('%s use time:  %s'%(function.__name__, str(end-start)))
		return result
	return function_timer


@fn_timer
def file_read(para_name, para_splitprecent=0.9):
    """
        Read rating matrix, and split the file into trian set and test set. 
        Return rating matrix and test data.
    :param para_name:           file name
    :param para_splitprecent:   the percent of train set and test set, default 0.9
    :return:                    rating matrix, test data, dict based on user and dict based on item
    user_dict = {
            0:  [(0, 5.0), (1, 2.0), ... ],
            1:  [(0, 1.0), (1, 0.0), ... ],
            ...
    }
    ===>    user_id: [(item_id, rating)...]
    item_dict = {    
        0:  [(0, 5.0), (1, 2.0), ... ],
        1:  [(0, 1.0), (1, 0.0), ... ],
        ...
    }
    ===>    item_id: [(user_id, rating)...]
    """
    test_data = []
    rate = pd.read_csv(para_name)
    del rate["timestamp"]
    user_num = max(rate['userId'])
    item_num = max(rate['movieId'])
    rate = rate.values
    rate_m = sparse.dok_matrix((user_num, item_num))
    user_dict = {}
    item_dict = {}
    for vec in rate:
        if rd.random() > para_splitprecent:                     # test data
            test_data.append([int(vec[0]-1), int(vec[1]-1), vec[2]])    
        else:                                                   # train data
            rate_m[int(vec[0]-1), int(vec[1]-1)] = vec[2]       # array start from 0  
            try:
                user_dict[int(vec[0]-1)].append((int(vec[1]-1), vec[2]))
            except:
                user_dict[int(vec[0]-1)] = []
                user_dict[int(vec[0]-1)].append((int(vec[1]-1), vec[2]))
            try:
                item_dict[int(vec[1]-1)].append((int(vec[0]-1), vec[2]))
            except:
                item_dict[int(vec[1]-1)] = []
                item_dict[int(vec[1]-1)].append((int(vec[0]-1), vec[2]))
    return rate_m, test_data, user_dict, item_dict
    

@fn_timer
def file_read_2(para_name, para_splitprecent=0.9, para_max_user=100000, para_max_item=200000):
    """
        When file is to large to load in memory, use this function.
        Read rating matrix, and split the file into trian set and test set. 
        Return rating matrix and test data.
    :param para_name:           file name
    :param para_splitprecent:   the percent of train set and test set, default 0.9
    :return:                    rating matrix and test data
    user_dict = {
        0:  [(0, 5.0), (1, 2.0), ... ],
        1:  [(0, 1.0), (1, 0.0), ... ],
            ...
    }
    ===>    user_id: [(item_id, rating)...]
    item_dict = {    
        0:  [(0, 5.0), (1, 2.0), ... ],
        1:  [(0, 1.0), (1, 0.0), ... ],
        ...
    }
    ===>    item_id: [(user_id, rating)...]
    """
    test_data = []
    rate_m = sparse.dok_matrix((para_max_user, para_max_item))
    with open(para_name) as f:
        f.readline()
        while True:
            tmp = f.readline().split(',')
            if len(tmp) < 2:
                break
            vec = [int(tmp[0]), int(tmp[1]), float(tmp[2])]         
            if rd.random() > para_splitprecent:                             # test data
                test_data.append([int(vec[0]-1), int(vec[1]-1), vec[2]])    
            else:                                                           # train data
                rate_m[int(vec[0]-1), int(vec[1]-1)] = vec[2]               # array start from 0
                if user_dict[int(vec[0]-1)] == None:
                    user_dict[int(vec[0]-1)] = []
                if item_dict[int(vec[1]-1)] == None:
                    item_dict[int(vec[1]-1)] = []
                user_dict[int(vec[0]-1)].append((int(vec[1]-1), vec[2]))
                item_dict[int(vec[1]-1)].append((int(vec[0]-1), vec[2]))
    return rate_m, test_data, user_dict, item_dict


@fn_timer
def get_dict_user(para_m):
    """
        Return a dict based on user.
        Record rating information of one user.
        user = {
            0:  [(0, 5.0), (1, 2.0), ... ],
            1:  [(0, 1.0), (1, 0.0), ... ],
            ...
        }
        user_id: [(item_id, rating)...]
    :param para_m:      rating matrix
    :return:            a dict     
    """
    num_user, num_item = para_m.shape
    user = {}
    for i in range(num_user):
        user[i] = []
        for j in range(num_item):
            if para_m[i,j] > 0:
                user[i].append((j, para_m[i,j]))
    return user


@fn_timer
def get_dict_item(para_m):
    """
        Return a dict based on item.
        Record rating information of one item.
        item = {    
            0:  [(0, 5.0), (1, 2.0), ... ],
            1:  [(0, 1.0), (1, 0.0), ... ],
            ...
        }
        item_id: [(user_id, rating)...]
    :param para_m:      rating matrix
    :return:            a dict     
    """
    num_user, num_item = para_m.shape
    item = {}
    for j in range(num_item):
        item[j] = []
        for i in range(num_user):
            if para_m[i,j] > 0:
                item[j] = (i, para_m[i,j])
    return item


@fn_timer
def get_dict_i_u(para_m):
    """
        Return a dict based on item and dict.
        Record rating information of one item.
        item = {    
            0:  [(0, 5.0), (1, 2.0), ... ],
            1:  [(0, 1.0), (1, 0.0), ... ],
            ...
        }
        item_id: [(user_id, rating)...]
        user = {
            0:  [(0, 5.0), (1, 2.0), ... ],
            1:  [(0, 1.0), (1, 0.0), ... ],
            ...
        }
        user_id: [(item_id, rating)...]        
    :param para_m:      rating matrix
    :return:            dict of item and user   
    """    
    num_user, num_item = para_m.shape
    item = {}
    user = {}
    for j in range(num_item):
        for i in range(num_user):
            if para_m[i,j] > 0:
                if j not in item.keys():
                    item[j] = []
                if i not in user.keys():
                    user[i] = []
                item[j].append((i, para_m[i,j]))
                user[i].append((j, para_m[i,j])) 
    return item, user


# @fn_timer
def loss_rmse(para_hat, para_true, skip=0):
    """
        The RMSE loss.
        The format of input vector:
            user_id, item_id, rate
    :param para_hat:    estimated value
    :param para_true:   true value
    :return:            the rmse loss
    """
    loss = 0
    n = len(para_hat)
    for ii in range(n):
        loss += pow(para_hat[ii][2] - para_true[ii][2], 2)
    return loss/(n-skip)


# @fn_timer
def loss_rmae(para_hat, para_true, skip=0):
    """
        The RMSE loss.
        The format of input vector:
            user_id, item_id, rate
    :param para_hat:    estimated value
    :param para_true:   true value
    :return:            the rmse loss
    """
    loss = 0
    n = len(para_hat)
    for ii in range(n):
        loss += abs(para_hat[ii][2] - para_true[ii][2])
    return loss/(n-skip)
