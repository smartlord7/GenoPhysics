from types import FunctionType


def get_var_index(var):
    return int(var[1:])


def generate_vars(n_vars: int,
                  suffix: str = 'X'):
    vars_set = []

    for i in range(n_vars):
        vars_set.append(suffix + str(i))

    return vars_set


def is_var(name):
    """Test: is name a variable?"""
    return isinstance(name, str) and (name[0] == 'X') and (name[1:].isdigit())


def individual_size(indiv):
    """ Number of nodes of an individual."""
    if not isinstance(indiv, list):
        return 1
    else:
        return 1 + sum(map(individual_size, indiv[1:]))


# Interpreter. FGGP, algorithm 3.1 - pg.25
def interpreter(individual, variables):
    value = None

    if isinstance(individual, list):
        try:
            func = eval(individual[0])
            if isinstance(func, FunctionType) and (len(individual) > 1):
                # Function: evaluate
                value = func(*[interpreter(arg, variables) for arg in individual[1:]])
            else:
                # Macro: don't evaluate arguments
                value = individual
        except Exception:
            print('error')

    elif isinstance(individual, (float, int)):
        # It's a constant
        value = individual
    elif is_var(individual):
        # It's a variable
        index = get_var_index(individual)
        value = variables[index]  # binding value
    else:
        # Terminal 0-ary function: execute
        value = individual()

    return value


def is_float(string):
    try:
        float(string)

        return True
    except ValueError:

        return False


def tree_to_inline_expression(tree, decimal_places: int = 5):
    if isinstance(tree, list):
        operator = ' ' + eval(tree[0]).__annotations__['symbol'] + ' '
        operands = [tree_to_inline_expression(subtree) for subtree in tree[1:]]
        operands = [('%.' + str(decimal_places) + 'f') % float(operand) if is_float(operand) else operand for operand in operands]

        return f'({operator.join(operands)})'
    else:
        return str(tree)
