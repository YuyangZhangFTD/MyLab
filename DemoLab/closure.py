def f1(x):
    z = x
    def f2(y):
        x = z + 5
        return x+y
    return f2


f = f1(2)
print(f(3))
print(f(4))
