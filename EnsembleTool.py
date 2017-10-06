import RecTool as rt
import random
from scipy import sparse 


def get_sub_set(para_m, para_k, para_percent):
    para_m = sparse.dok_matrix(para_m)
    user_num, item_num = para_m.shape
    res_list = [sparse.lil_matrix((user_num, item_num)) for __ in range(para_k)]
    for (u, i), r in para_m.items():
        for num in range(para_k):
            if random.random() > para_percent:
                continue
            res_list[num][u, i] = r
    return res_list
