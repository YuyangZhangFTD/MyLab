import numpy as np
from functools import reduce


class IREvaluation(object):
    """[summary]
        Evaluations in IR:
            1. NDCG
            2. ERR
    """

    def __init__(self, grade_max=4, grade_min=0):
        self.ranked_list = []
        self.ideal_list = []
        self.grade_list = []
        self.grade_max = grade_max
        self.grade_min = grade_min = 0
        self.grade_satisfaction_probability = {
            i: (2 ** i - 1) / 2 ** (self.grade_max - self.grade_min)
            for i in range(self.grade_max - self.grade_min + 1)
        }

    def DCG(self, ranked_list):
        num = len(ranked_list)
        normalized_list = [
            x - self.grade_min for x in ranked_list] if self.grade_min != 0 else ranked_list
        value = 2 ** normalized_list[0] - 1
        for i in range(1, num):
            value += (2 ** normalized_list[i] - 1) / np.log2(i + 2)
        return value

    def NDCG(self, evaluted_list):
        actual_list = evaluted_list.copy()
        evaluted_list.sort(reverse=True)
        ideal_list = evaluted_list
        return self.DCG(actual_list) / self.DCG(ideal_list)

    def average_NDCG(self, test_set):
        num = len(test_set)
        sum_NDCG = 0
        for lst in test_set:
            sum_NDCG += self.NDCG(lst)
        return sum_NDCG / num

    def ERR(self, ranked_list):
        rank = 1
        value = 0
        prev = []
        for grade in ranked_list:
            value += reduce(lambda x, y: x * y, [1] + [1 - k for k in prev]) * \
                self.grade_satisfaction_probability[grade] / rank
            prev.append(self.grade_satisfaction_probability[grade])
            rank += 1
        return value

    def average_ERR(self, test_set):
        num = len(test_set)
        sum_ERR = 0
        for lst in test_set:
            sum_ERR += self.ERR(lst)
        return sum_ERR / num


if __name__ == "__main__":
    e = IREvaluation(grade_max=4)
    # print(e.average_NDCG([
    #     [2, 3, 1, 4, 1, 2, 2, 3],
    #     [1, 2, 3, 3, 4, 0, 0],
    #     [2, 0, 1, 0, 0]
    # ]))
    print(e.average_ERR(
        [[2, 3, 1, 4, 1, 2, 2, 3],
         [1, 2, 3, 3, 4, 0, 0]]))
