class Variable(object):
    def __init__(self, name, index):

        # variable name
        self.name = name

        # variable index
        self.index = index

        # 0 for init
        self.coef = 0

        # -1 for init
        self.next = []

        # -1 for init
        #  0 for leq
        #  1 for <
        #  2 for geq
        #  3 for >
        #  4 for =
        self.comp = -1

    def __add__(self, other):
        self.next.append((other.name, other.index, other.coef))
        return self

    def __sub__(self, other):
        other.coef *= -1
        self.next.append(other.index)
        return self

    def __mul__(self, other):
        self.coef = other
        return self


if __name__ == "__main__":
    a = Variable("a", (0, 1))
    b = Variable("b", (0, 2))
    c = Variable("c", (0, 3))
    a * 2 + b * 3 + c * 5
    print("a")
    print(a.coef)
    print(a.next)
    print("b")
    print(b.coef)
    print(b.next)
    print(b.comp)
    print("c")
    print(c.coef)
    print(c.next)
