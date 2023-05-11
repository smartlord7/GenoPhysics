import sympy as sp
from typing import Any
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
        except Exception as e:
            print('Unknown error')

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


def tree_to_inline_expression(tree: list, decimal_places: int = 5) -> tuple[str, Any]:
    non_simplified_expr = tree_to_inline_expression_(tree)
    expr = sp.sympify(non_simplified_expr)
    simplified_expr = sp.simplify(expr)
    simplified_expr = sp.factor(simplified_expr)
    simplified_expr = sp.N(simplified_expr, decimal_places)

    return non_simplified_expr, simplified_expr


def tree_to_inline_expression_(tree):
    if isinstance(tree, list):
        operator = ' ' + eval(tree[0]).__annotations__['symbol'] + ' '
        operands = [tree_to_inline_expression_(subtree) for subtree in tree[1:]]

        return f'({operator.join(operands)})'
    else:
        return str(tree)


def simplify_expression(expr):
    """
    Simplify a math expression defined by functions and variables.
    """
    # Split the expression into the function and the arguments
    func_name, args = expr.split('(', 1)
    args = args[:-1]  # Remove the closing parenthesis
    # Split the arguments into a list of arguments
    args_list = []
    arg = ''
    depth = 0
    for c in args:
        if c == ',' and depth == 0:
            args_list.append(arg)
            arg = ''
        else:
            arg += c
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
    args_list.append(arg)
    # Simplify the arguments recursively
    args_simp = [simplify_expression(a) if '(' in a else a for a in args_list]
    # Simplify the function
    if func_name == 'function_wrappers.add_w':
        return '({} + {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.sub_w':
        return '({} - {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.mult_w':
        return '({} * {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.div_prot_w':
        return '({} / {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.sin_w':
        return 'sin({})'.format(args_simp[0])
    elif func_name == 'function_wrappers.cos_w':
        return 'cos({})'.format(args_simp[0])
    elif func_name == 'function_wrappers.exp_w':
        return 'exp({})'.format(args_simp[0])
    elif func_name == 'x':
        return 'x{}'.format(int(args_simp[0]))
    else:
        raise ValueError('Unknown function: {}'.format(func_name))

