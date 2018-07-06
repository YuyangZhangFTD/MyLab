import numpy as np
import numpy.linalg as la
from Variable import Variable


def quick_sum(vars_list=list()):
    if len(vars_list) == 0:
        print("No available variables")
        return None
    tmp = vars_list[0]
    for i in range(1, len(vars_list)):
        tmp += vars_list[i]
    coefficient_list = tmp.next
    coefficient_list.inset(0, (tmp.name, tmp.index, tmp.coef))
    return coefficient_list


class LpSolver(object):
    """
    standard form:
    min cx
    s.t. Ax >= b
    """

    def __init__(self):
        # objective type
        # 0 for init
        # 1 for min (standard)
        # 2 for max
        self.obj_type = 0

        # name - variables_dict
        # name(variables_dict) : Variable
        self.variables_dict = dict()

        # name : constraints_dict
        # name(constraints_dict) : [coefficient_list, comp_sign, comp_value]
        self.constraints_dict = dict()

    def add_variables(self, name=None, variables_index=list()):

        if len(variables_index) == 0:
            print("No variables added")
            return False

        if name:
            print("No variables name")

        var = dict()
        for index in variables_index:
            var[tuple(index)] = Variable(name=name, index=index)

        self.var_dict[name] = var
        return var

    def add_constraint(self, name=None, constraint_index=list()):
        pass

    def add_constraints(self):
        pass

    def set_objective(self, obj, type="min"):

        if type == "min" or obj == 1:
            self.obj_type = 1
        elif type == "max" or obj == 2:
            self.obj_type = 2
        else:
            print("Wrong objective type")

    def pre_check(self):
        if self.obj_type == 0:
            print("please set objective type")
            return False
        elif self.obj_type == 2:
            self.c *= -1
            self.obj_type = 1

        if not (
            self.c.shape[0] == 1 and
            self.b.shape[1] == 1 and
            self.c.shape[1] == self.A.shape[1] and
            self.A.shape[0] == self.b.shape[0]
        ):
            print("Wrong data matrix input")
            return False

        return True


if __name__ == "__main__":
    pass
