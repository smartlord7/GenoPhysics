def add_w(x, y):
    return x + y


def mult_w(x, y):
    return x * y


def sub_w(x, y):
    return x - y


def div_prot_w(x, y):
    if abs(y) <= 1e-3:
        return 1
    else:
        return x / y
