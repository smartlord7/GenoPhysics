def symbol(value):
    def decorator(func):
        func.__annotations__['symbol'] = value

        return func

    return decorator


@symbol('+')
def add_w(x, y):
    return x + y


@symbol('*')
def mult_w(x, y):
    return x * y


@symbol('-')
def sub_w(x, y):
    return x - y


@symbol('/')
def div_prot_w(x, y):
    if abs(y) <= 1e-3:
        return 1
    else:
        return x / y
